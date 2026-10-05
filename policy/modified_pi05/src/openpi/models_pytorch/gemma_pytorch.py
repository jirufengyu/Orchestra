from typing import Literal

import logging

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

from openpi.models_pytorch.moe_action_expert import MoEActionExpert


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        *,
        use_action_expert: bool = True,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        # ── MoE parameters ──
        use_moe: bool = False,
        moe_task_names: list[str] | None = None,
        moe_use_shared_expert: bool = True,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.use_action_expert = bool(use_action_expert)
        self.use_moe = use_moe

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        # Important: image features are projected to `vision_config.projection_dim` and then concatenated
        # with language token embeddings. This must match the text hidden size, otherwise embed_prefix()
        # will fail when concatenating image + language embeddings (e.g. gemma_300m has width=1024).
        vlm_config_hf.vision_config.projection_dim = vlm_config.width
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)

        if self.use_action_expert:
            action_expert_config_hf = CONFIG_MAPPING["gemma"](
                head_dim=action_expert_config.head_dim,
                hidden_size=action_expert_config.width,
                intermediate_size=action_expert_config.mlp_dim,
                num_attention_heads=action_expert_config.num_heads,
                num_hidden_layers=action_expert_config.depth,
                num_key_value_heads=action_expert_config.num_kv_heads,
                vocab_size=257152,
                hidden_activation="gelu_pytorch_tanh",
                torch_dtype="float32",
                use_adarms=use_adarms[1],
                adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
            )

            if use_moe:
                # ── MoE mode: create multiple task-specific experts ──
                if moe_task_names is None or len(moe_task_names) == 0:
                    raise ValueError("use_moe=True requires non-empty moe_task_names")
                self.moe_expert = MoEActionExpert(
                    expert_config=action_expert_config_hf,
                    task_names=moe_task_names,
                    use_shared_expert=moe_use_shared_expert,
                )
                # Keep gemma_expert as None – all access goes through moe_expert
                self.gemma_expert = None
                logging.info(
                    f"Created MoE action expert with {len(moe_task_names)} tasks: {moe_task_names}"
                )
            else:
                # ── Standard single-expert mode ──
                self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
                self.gemma_expert.model.embed_tokens = None
                self.moe_expert = None

            self.suffix_in_proj = None
            self.suffix_out_proj = None
        else:
            # no-expert 模式：不创建 Gemma expert（参数大头），而用轻量投影桥接维度：
            # suffix_in:  expert_width -> vlm_width （喂给 paligemma LM）
            # suffix_out: vlm_width -> expert_width（保持 suffix_out / 下游 head 的维度不变）
            self.gemma_expert = None
            self.moe_expert = None
            self.suffix_in_proj = nn.Linear(action_expert_config.width, vlm_config.width, bias=False)
            self.suffix_out_proj = nn.Linear(vlm_config.width, action_expert_config.width, bias=False)

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _get_active_expert(self, task_id=None) -> GemmaForCausalLM | None:
        """Return the active ``GemmaForCausalLM`` expert based on *task_id*.

        - **MoE mode** (``use_moe=True``): delegates to ``self.moe_expert``.
        - **Single-expert mode**: returns ``self.gemma_expert`` (ignores *task_id*).
        - **No-expert mode**: returns ``None``.
        """
        if self.moe_expert is not None:
            return self.moe_expert.get_expert_for_task(task_id)
        return self.gemma_expert  # may be None in no-expert mode

    def _get_active_expert_model(self, task_id=None):
        """Return the underlying ``GemmaModel`` (i.e. ``.model``) of the active expert.

        This is the object whose ``.layers`` are used in the layer-wise interleaved
        computation.
        """
        expert = self._get_active_expert(task_id)
        if expert is None:
            return None
        return expert.model

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_image_with_memory(
        self,
        images: torch.Tensor,
        frame_mask: torch.Tensor | None,
        *,
        temporal_pe: torch.Tensor | None,
        temporal_interval: int = 4,
        drop_past_after_layer: int | None = None,
    ):
        """Encode a short-horizon frame window with MEM-style in-ViT temporal attention."""
        if images.ndim != 5:
            raise ValueError(f"images must have shape [B, K, C, H, W] or [B, K, H, W, C], got {images.shape}")
        bsize, k = images.shape[:2]
        flat = images.reshape(bsize * k, *images.shape[2:])
        if flat.shape[1] != 3 and flat.shape[-1] == 3:
            flat = flat.permute(0, 3, 1, 2)
        if k == 1:
            return self.embed_image(flat)
        window_pe = temporal_pe[-k:] if temporal_pe is not None else None
        return self.paligemma.model.get_image_features(
            flat,
            num_frames=k,
            frame_mask=frame_mask,
            temporal_pe=window_pe,
            temporal_interval=temporal_interval,
            drop_past_after_layer=drop_past_after_layer,
        )

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        task_id: str | int | None = None,
        return_intermediate_layer_idx: int | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]

        # Resolve the active expert model for this forward pass
        active_expert = self._get_active_expert(task_id)
        active_expert_model = active_expert.model if active_expert is not None else None
        intermediate_suffix_output = None

        if not self.use_action_expert:
            def _cast_attention_mask(mask: torch.Tensor | None, *, dtype: torch.dtype):
                # SDPA 路径要求 bias(mask) dtype 与 query dtype 一致，否则报：
                # RuntimeError: invalid dtype for bias - should match query's dtype
                if mask is None:
                    return None
                if mask.dtype == dtype:
                    return mask
                return mask.to(dtype=dtype)

            if inputs_embeds is None:
                raise ValueError("inputs_embeds must be provided (list[prefix_embs, suffix_embs])")
            if len(inputs_embeds) != 2:  # noqa: PLR2004
                raise ValueError("inputs_embeds must be a list of length 2: [prefix_embs, suffix_embs]")

            prefix_embs, suffix_embs = inputs_embeds
            if suffix_embs is not None:
                if self.suffix_in_proj is None or self.suffix_out_proj is None:
                    raise RuntimeError("no-expert mode expects suffix_{in,out}_proj to be initialized")
                suffix_embs_vlm = self.suffix_in_proj(suffix_embs)
            else:
                suffix_embs_vlm = None

            if prefix_embs is None and suffix_embs_vlm is None:
                raise ValueError("At least one of prefix_embs / suffix_embs must be non-None")

            if prefix_embs is None:
                attention_mask_ = _cast_attention_mask(attention_mask, dtype=suffix_embs_vlm.dtype)
                out = self.paligemma.language_model.forward(
                    inputs_embeds=suffix_embs_vlm,
                    attention_mask=attention_mask_,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
                )
                suffix_out_vlm = out.last_hidden_state
                suffix_out = self.suffix_out_proj(suffix_out_vlm)
                return [None, suffix_out], out.past_key_values

            if suffix_embs_vlm is None:
                attention_mask_ = _cast_attention_mask(attention_mask, dtype=prefix_embs.dtype)
                out = self.paligemma.language_model.forward(
                    inputs_embeds=prefix_embs,
                    attention_mask=attention_mask_,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
                )
                return [out.last_hidden_state, None], out.past_key_values

            # both present: concat -> paligemma LM -> split -> project suffix back
            prefix_len = prefix_embs.shape[1]
            cat_embs = torch.cat([prefix_embs, suffix_embs_vlm], dim=1)
            attention_mask_ = _cast_attention_mask(attention_mask, dtype=cat_embs.dtype)
            out = self.paligemma.language_model.forward(
                inputs_embeds=cat_embs,
                attention_mask=attention_mask_,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            hs = out.last_hidden_state
            prefix_output = hs[:, :prefix_len, :]
            suffix_out_vlm = hs[:, prefix_len:, :]
            suffix_output = self.suffix_out_proj(suffix_out_vlm)
            return [prefix_output, suffix_output], out.past_key_values

        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = active_expert_model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
                output_hidden_states=return_intermediate_layer_idx is not None,
            )
            intermediate_suffix_output = None
            if return_intermediate_layer_idx is not None:
                intermediate_suffix_output = suffix_output.hidden_states[return_intermediate_layer_idx + 1]
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, active_expert_model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers
            intermediate_suffix_output = None

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(active_expert_model, "gradient_checkpointing")
                and active_expert_model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(active_expert_model, "gradient_checkpointing"):
                if not active_expert_model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    active_expert_model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(active_expert_model, 'gradient_checkpointing')}"
                )
                if hasattr(active_expert_model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {active_expert_model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Capture the active expert model for use inside the closure
            _active_expert_model = active_expert_model

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, _active_expert_model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                if layer_idx == return_intermediate_layer_idx:
                    intermediate_suffix_output = inputs_embeds[1]

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        if return_intermediate_layer_idx is not None:
            return [prefix_output, suffix_output], prefix_past_key_values, intermediate_suffix_output
        return [prefix_output, suffix_output], prefix_past_key_values
