import torch
from torch import nn
import torch.nn.functional as F
import openpi.models.gemma as _gemma
from openpi.models_pytorch.value_function_pytorch import Pi0ValueFunction
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

class Pi0QFunction(Pi0ValueFunction):
    """
    Q-Function 变体：输入 Observation 和 Action，输出 Return。
    用于对采样出的轨迹进行打分 (Test-time Re-ranking)。
    """
    def __init__(self, config, num_bins=201):
        super().__init__(config, num_bins)
        
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        
        # Action Projection: 将动作维度映射到模型宽度
        # 如果 config 中没有 action_dim，默认为 32 (Pi0 常用值)
        self.action_dim = getattr(config, "action_dim", 32)
        self.action_proj = nn.Linear(self.action_dim, action_expert_config.width)

    def embed_suffix(self, state, actions, bsize, device):
        """
        处理 Suffix: State + Actions
        """
        embs = []
        pad_masks = []
        att_masks = []

        # 1. State Embedding (Always use state for Q-function if available)
        use_state = not self.pi05
        if use_state:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)
            embs.append(state_emb[:, None, :]) # [B, 1, W]
            
            # Mask for State
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            # Att Mask = 1 (Separates Prefix from Suffix)
            att_masks.append(1)
        else:
            # Fallback to query token if no state (less common for Q-func)
            expert_input = self.value_query.expand(bsize, -1, -1).to(dtype=self.state_proj.weight.dtype)
            embs.append(expert_input)
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks.append(1)

        # 2. Action Embedding
        # actions shape: [B, T, D]
        if actions.shape[-1] != self.action_dim:
             # 如果维度不匹配，尝试简单 padding 或报错 (这里假设输入已经是正确的维度)
             # 对于 Pi0, 有时输入是 14 维，需要 pad 到 32
             if actions.shape[-1] < self.action_dim:
                 padding = torch.zeros(
                     actions.shape[:-1] + (self.action_dim - actions.shape[-1],), 
                     device=actions.device, dtype=actions.dtype
                 )
                 actions = torch.cat([actions, padding], dim=-1)

        def action_proj_func(actions):
            return self.action_proj(actions)

        action_embs = self._apply_checkpoint(action_proj_func, actions) # [B, T, W]
        embs.append(action_embs)

        # Mask for Actions
        T = action_embs.shape[1]
        pad_masks.append(torch.ones(bsize, T, dtype=torch.bool, device=device))
        
        # Att Mask = 0 (Same block as State, so full attention within Suffix)
        att_masks.extend([0] * T)

        # Concat
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self,
        observation,
        actions,
        return_probs: bool = False,
        *,
        return_ensemble_stats: bool = False,
        return_score_stats: bool = False,
        score_bins_min: float = -1.0,
        score_bins_max: float = 0.0,
    ):
        """
        Args:
            observation: Pi0 Observation Dict
            actions: [B, T, Action_Dim]
        备注：
            - 当 return_probs=True 且 return_ensemble_stats=True 时，返回的是 (probs_mean, logits_var)。
        """
        # 1. Preprocess
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        # 2. Embed Prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # 3. Embed Suffix (State + Actions)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, actions, bsize, device)

        # Cast to bfloat16 if needed
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # 4. Masks
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # 5. Forward
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, None],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids
        )

        # 6. Prediction
        # Take the embedding of the last token (last action step)
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
            out = (probs, logits_var) if return_ensemble_stats else probs
        else:
            if not (return_ensemble_stats or return_score_stats):
                return logits_mean
            out = (logits_mean, logits_var) if return_ensemble_stats else logits_mean

        if not return_score_stats:
            return out

        # score stats（ensemble 维度）：将每个 head 的 logits 解码成期望值
        if self.num_value_heads == 1:
            score = self.logits_to_expected_value(
                logits_mean, bins_min=score_bins_min, bins_max=score_bins_max
            )
            score_mean = score
            score_var = torch.zeros_like(score)
        else:
            logits_per_head = []
            for head in self._iter_value_heads():
                logits_per_head.append(head(value_embedding))
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
