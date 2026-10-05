import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks, get_safe_dtype

class Pi0ValueFunction(nn.Module):
    def __init__(self, config, num_bins=201):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.num_bins = num_bins

        # Value Function 通常使用较小的 VLM backbone (如 Gemma 3 670M)
        # config.paligemma_variant 和 config.action_expert_variant 应该在传入前配置好
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # 是否启用 action expert（默认 True 以保持向后兼容；可在 config 中关闭以移除大块 expert 参数）
        vf_use_action_expert = getattr(config, "vf_use_action_expert", None)
        if vf_use_action_expert is None:
            vf_use_action_expert = True

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            # VF 不需要 adaRMS (用于 flow matching 的时间调节)，所以设为 False
            use_adarms=[False, False],
            use_action_expert=bool(vf_use_action_expert),
            precision=config.dtype,
        )

        # 状态投影层 (如果启用状态输入)
        # 在 VF 中，我们总是希望利用状态信息，如果没有状态，我们使用一个可学习的 Query
        self.state_proj = nn.Linear(32, action_expert_config.width)
        
        # 如果不使用状态 (pi05模式) 或者作为补充，我们可以使用一个可学习的 Query Token
        self.value_query = nn.Parameter(torch.randn(1, 1, action_expert_config.width))

        # ===== Value heads (支持可选 ensemble，默认向后兼容单头) =====
        #
        # 说明：
        # - 默认 num_value_heads=1：参数名与行为保持不变（仍然只有 self.value_head），
        #   以确保旧 checkpoint 在 strict-load 下不会缺失 key。
        # - 当 num_value_heads>1：额外创建 self.value_heads_extra（ModuleList），
        #   forward 默认仍返回 mean logits（旧调用不受影响），可选返回均值/方差统计。
        self.num_value_heads = int(
            getattr(config, "num_value_heads", None)
            or getattr(config, "value_num_heads", None)
            or getattr(config, "value_head_ensemble_size", None)
            or getattr(config, "value_ensemble_size", None)
            or 1
        )
        if self.num_value_heads < 1:
            raise ValueError(f"num_value_heads must be >= 1, got {self.num_value_heads}")

        # Dropout：为了让 ensemble heads 不容易学到同一模式，可以在每个 head 前对 embedding 做随机 dropout。
        # 默认 p=0.0（完全兼容旧行为）。如需开启可在 config 里设置：
        # - value_head_dropout_p / value_dropout_p
        # - value_head_dropout_in_eval（可选：推理时也启用 MC-dropout 风格的不确定性）
        self.value_head_dropout_p = float(
            getattr(config, "value_head_dropout_p", None)
            or getattr(config, "value_dropout_p", None)
            or 0.0
        )
        if not (0.0 <= self.value_head_dropout_p < 1.0):
            raise ValueError(
                f"value_head_dropout_p must be in [0, 1), got {self.value_head_dropout_p}"
            )
        self.value_head_dropout_in_eval = bool(
            getattr(config, "value_head_dropout_in_eval", False)
        )

        # 输出头：映射到离散的价值分箱（第 0 个 head 保持原名 value_head，便于兼容旧 ckpt）
        self.value_head = nn.Linear(action_expert_config.width, num_bins)
        self.value_heads_extra = nn.ModuleList()
        if self.num_value_heads > 1:
            for _ in range(self.num_value_heads - 1):
                self.value_heads_extra.append(nn.Linear(action_expert_config.width, num_bins))

        torch.set_float32_matmul_precision("high")
        
        # Gradient Checkpointing 设置
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        # no-expert 模式下 gemma_expert 可能为 None
        if getattr(self.paligemma_with_expert, "gemma_expert", None) is not None:
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _iter_value_heads(self) -> list[nn.Module]:
        # 保证顺序稳定：value_head + extras
        if self.num_value_heads == 1:
            return [self.value_head]
        return [self.value_head, *list(self.value_heads_extra)]

    def _maybe_dropout_value_embedding(self, value_embedding: Tensor) -> Tensor:
        """
        对 value_embedding 做随机 dropout。
        - 默认只在训练时生效
        - 如 value_head_dropout_in_eval=True，则 eval() 下也会随机 dropout（MC-dropout 风格）
        """
        if self.value_head_dropout_p <= 0.0:
            return value_embedding
        training_flag = self.training or self.value_head_dropout_in_eval
        # 用 functional dropout，避免依赖模块的 training 状态；每次调用都会采样新的 mask（随机的）。
        return F.dropout(value_embedding, p=self.value_head_dropout_p, training=training_flag)

    def _value_logits_from_embedding(
        self,
        value_embedding: Tensor,
        *,
        return_ensemble_stats: bool = False,
    ):
        """
        从 value embedding 计算 logits。

        Returns:
            - 如果 return_ensemble_stats=False:
                logits_mean: [B, num_bins]
            - 如果 return_ensemble_stats=True 且 num_value_heads>1:
                (logits_mean, logits_var): 两者均为 [B, num_bins]
              若 num_value_heads==1，则 logits_var 为全 0（同 shape）。
        """
        if self.num_value_heads == 1:
            emb = self._maybe_dropout_value_embedding(value_embedding)
            logits = self.value_head(emb)  # [B, num_bins]
            if not return_ensemble_stats:
                return logits
            return logits, torch.zeros_like(logits)

        logits_per_head = []
        for head in self._iter_value_heads():
            # 每个 head 都单独做一次 dropout -> mask 独立，增强多样性
            emb = self._maybe_dropout_value_embedding(value_embedding)
            logits_per_head.append(head(emb))
        logits_stack = torch.stack(logits_per_head, dim=0)  # [H, B, num_bins]
        logits_mean = logits_stack.mean(dim=0)
        if not return_ensemble_stats:
            return logits_mean
        logits_var = logits_stack.var(dim=0, unbiased=False)
        return logits_mean, logits_var

    @staticmethod
    def logits_to_expected_value(
        logits: Tensor,
        *,
        bins_min: float = -1.0,
        bins_max: float = 0.0,
    ) -> Tensor:
        """
        将 logits([B, num_bins]) 转成期望值 E[bin]([B])。
        bins 默认与 policy.py 中一致：[-1.0, 0.0]。
        """
        probs = F.softmax(logits, dim=-1)
        bins = torch.linspace(bins_min, bins_max, logits.shape[-1], device=logits.device, dtype=probs.dtype)
        return (probs * bins[None, :]).sum(dim=-1)

    def _preprocess_observation(self, observation, train=False):
        # 复用预处理逻辑
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """处理图像和文本 (Prefix) - 与 Pi0 策略模型相同"""
        embs = []
        pad_masks = []
        att_masks = []

        # 1. Image Embeddings
        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)
            
            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # 2. Language Embeddings
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, bsize, device):
        """
        处理 Suffix (Expert 输入)。
        不同于 Policy 模型，VF 没有 noisy actions 和 time。
        我们输入 State (如果可用) 或者 Value Query Token。
        Expert 将通过 Cross-Attention 关注 Prefix (图像+文本) 来预测价值。
        """
        embs = []
        pad_masks = []
        att_masks = []

        # 尝试使用 State Embedding
        use_state = not self.pi05 # Pi06 通常包含 State
        
        if use_state:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)
            # state_emb shape: [B, width] -> [B, 1, width]
            expert_input = state_emb[:, None, :]
        else:
            # 如果不使用 State，使用可学习的 Value Query Token
            # [1, 1, width] -> [B, 1, width]
            expert_input = self.value_query.expand(bsize, -1, -1)
            # 确保 dtype 匹配
            expert_input = expert_input.to(dtype=self.state_proj.weight.dtype)

        embs.append(expert_input)
        
        # Masking
        expert_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(expert_mask)
        
        # Attention Mask: 1 表示这个 token 不被 prefix 里的 token 看到 (这是 Expert 的输入)
        # 但是它自己可以看到 prefix (通过 Causal Masking 的下三角性质 + Cross Attention)
        att_masks += [1] 

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self,
        observation,
        actions=None,
        return_probs: bool = False,
        *,
        return_ensemble_stats: bool = False,
        return_score_stats: bool = False,
        score_bins_min: float = -1.0,
        score_bins_max: float = 0.0,
    ):
        """
        前向传播计算价值分布。
        Args:
            observation: 包含图像、文本指令、状态的观测数据
            actions: 不使用，为了兼容 Q-function 接口
            return_probs: 是否返回 softmax 后的概率 (Inference 用)
            return_ensemble_stats: 若为 True，返回 (logits_mean, logits_var)
            return_score_stats: 若为 True，额外返回由 bins 解码后的 score 均值/方差（ensemble 维度）
        Returns:
            默认：logits_mean: [B, num_bins]
            若 return_ensemble_stats=True： (logits_mean, logits_var)
            若 return_score_stats=True： 额外返回 score_stats dict（不改变 logits 返回类型的兼容性）
        备注：
            - 当 return_probs=True 且 return_ensemble_stats=True 时，返回的是 (probs_mean, logits_var)。
        """
        # 1. 预处理
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        # 2. Embed Prefix (PaliGemma)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # 3. Embed Suffix (Gemma Expert)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, bsize, device)

        # 统一 dtype (如果需要)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # 4. 拼接 Masks
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # 5. 模型前向传播 (Backbone + Expert)
        # 我们不需要 adarms_cond，因为没有 time conditioning
        adarms_cond = None 

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, None], # No ADARMS for VF
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids
        )

        # 6. Value Prediction
        # suffix_out shape: [B, 1, hidden_dim] (因为 suffix 长度为 1)
        value_embedding = suffix_out[:, -1, :].to(dtype=torch.float32)

        if return_ensemble_stats or return_score_stats:
            logits_mean, logits_var = self._value_logits_from_embedding(
                value_embedding, return_ensemble_stats=True
            )
        else:
            logits_mean = self._value_logits_from_embedding(
                value_embedding, return_ensemble_stats=False
            )

        if return_probs:
            probs = F.softmax(logits_mean, dim=-1)
            if not (return_ensemble_stats or return_score_stats):
                return probs
            # 如果要 stats，则在 probs 基础上继续返回 stats（保持默认分支兼容）
            out = (probs, logits_var) if return_ensemble_stats else probs
        else:
            if not (return_ensemble_stats or return_score_stats):
                return logits_mean
            out = (logits_mean, logits_var) if return_ensemble_stats else logits_mean

        if not return_score_stats:
            return out

        # score stats：将每个 head 的 logits 解码成期望值，再对 ensemble 维度求均值/方差
        if self.num_value_heads == 1:
            score = self.logits_to_expected_value(
                logits_mean, bins_min=score_bins_min, bins_max=score_bins_max
            )  # [B]
            score_mean = score
            score_var = torch.zeros_like(score)
        else:
            logits_per_head = []
            for head in self._iter_value_heads():
                emb = self._maybe_dropout_value_embedding(value_embedding)
                logits_per_head.append(head(emb))
            logits_stack = torch.stack(logits_per_head, dim=0)  # [H, B, num_bins]
            scores = []
            for h in range(logits_stack.shape[0]):
                scores.append(
                    self.logits_to_expected_value(
                        logits_stack[h], bins_min=score_bins_min, bins_max=score_bins_max
                    )
                )
            score_stack = torch.stack(scores, dim=0)  # [H, B]
            score_mean = score_stack.mean(dim=0)
            score_var = score_stack.var(dim=0, unbiased=False)

        score_stats = {
            "score_mean": score_mean,
            "score_var": score_var,
            "score_std": torch.sqrt(torch.clamp(score_var, min=0.0)),
        }

        return out, score_stats