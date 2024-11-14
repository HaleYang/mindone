from typing import Any, Dict, List, Tuple, Optional, Union
import mindspore as ms
from mindspore import nn

from mindone.diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from mindone.diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from mindone.diffusers.models.attention import JointTransformerBlock, _chunked_feed_forward
from .attention_cache import Attention, JointAttnProcessor


class JointTransformerCacheBlock(JointTransformerBlock):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, context_pre_only=False, layer_idx=0):
        super().__init__(dim, num_attention_heads, attention_head_dim, context_pre_only)

        processor = JointAttnProcessor()

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim // num_attention_heads,
            heads=num_attention_heads,
            out_dim=attention_head_dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
            layer_idx=layer_idx,
        )


    def construct(self, hidden_states: ms.Tensor, encoder_hidden_states: ms.Tensor, temb: ms.Tensor):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (None,) * 5
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states



class SD3Transformer2DModelCache(SD3Transformer2DModel):
    def __init__(
        self, 
        sample_size: int = 128, 
        patch_size: int = 2, 
        in_channels: int = 16, 
        num_layers: int = 18, 
        attention_head_dim: int = 64, 
        num_attention_heads: int = 18, 
        joint_attention_dim: int = 4096, 
        caption_projection_dim: int = 1152, 
        pooled_projection_dim: int = 2048, 
        out_channels: int = 16, 
        pos_embed_max_size: int = 96,
    ):
        super().__init__(
            sample_size, 
            patch_size, 
            in_channels, 
            num_layers, 
            attention_head_dim, 
            num_attention_heads, 
            joint_attention_dim, 
            caption_projection_dim, 
            pooled_projection_dim, 
            out_channels, pos_embed_max_size
        )

        self.transformer_blocks = nn.CellList(
            [
                JointTransformerCacheBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.inner_dim,
                    context_pre_only=(i == num_layers - 1),
                    layer_idx=i
                ) 
                for i in range(self.config.num_layers)
            ]
        )

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
        pooled_projections: ms.Tensor = None,
        timestep: ms.Tensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
        cache_params: Tuple = None,
        if_skip: bool = False,
        delta_cache: ms.Tensor = None,
        delta_cache_hidden: ms.Tensor = None,
        use_cache: bool = False,
    ) -> Union[ms.Tensor, Transformer2DModelOutput, Tuple]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`ms.Tensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`ms.Tensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`ms.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `ms.Tensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `mindspore.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            cache_params (`Tuple`):
                A tuple of cache parameters which contains start cache layer id, step_stride, use cache layer nums, start use cache step 

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None and "scale" in joint_attention_kwargs:
            # weight the lora layers by setting `lora_scale` for each PEFT layer here
            # and remove `lora_scale` from each PEFT layer at the end.
            # scale_lora_layers & unscale_lora_layers maybe contains some operation forbidden in graph mode
            raise RuntimeError(
                f"You are trying to set scaling of lora layer by passing {joint_attention_kwargs['scale']=}. "
                f"However it's not allowed in on-the-fly model forwarding. "
                f"Please manually call `scale_lora_layers(model, lora_scale)` before model forwarding and "
                f"`unscale_lora_layers(model, lora_scale)` after model forwarding. "
                f"For example, it can be done in a pipeline call like `StableDiffusionPipeline.__call__`."
            )

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        (
            (encoder_hidden_states, hidden_states),
            delta_cache,
            delta_cache_hidden
        ) = self.forward_blocks(
            hidden_states, 
            encoder_hidden_states, 
            block_controlnet_hidden_states,
            temb,
            use_cache,
            if_skip,
            cache_params,
            delta_cache,
            delta_cache_hidden,
        )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config["patch_size"]
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            hidden_states.shape[0],
            height,
            width,
            patch_size,
            patch_size,
            self.out_channels,
        )
        # hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        hidden_states = hidden_states.transpose(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(
            hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size
        )

        if not return_dict:
            return (output, delta_cache, delta_cache_hidden) if use_cache else (output,)

        return Transformer2DModelOutput(sample=output)
    
    def forward_blocks_range(
        self, 
        hidden_states, 
        encoder_hidden_states, 
        block_controlnet_hidden_states, 
        temb, 
        start_idx, 
        end_idx,
    ):
        for index_block, block in enumerate(self.transformer_blocks[start_idx:end_idx]):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]
        
        return hidden_states, encoder_hidden_states

    def forward_blocks(
        self, 
        hidden_states, 
        encoder_hidden_states, 
        block_controlnet_hidden_states, 
        temb,
        use_cache,
        if_skip,
        cache_params,
        delta_cache,
        delta_cache_hidden,
    ):
        if not use_cache:
            hidden_states, encoder_hidden_states = self.forward_blocks_range(
                hidden_states,
                encoder_hidden_states,
                block_controlnet_hidden_states,
                temb,
                start_idx=0,
                end_idx=len(self.transformer_blocks)
            )
        else:
            # infer [0, cache_start)
            hidden_states, encoder_hidden_states = self.forward_blocks_range(
                hidden_states, 
                encoder_hidden_states,
                block_controlnet_hidden_states,
                temb,
                start_idx=0,
                end_idx=cache_params[0],
            )

            # infer [cache_start, cache_end)
            cache_end = cache_params[0] + cache_params[2]
            hidden_states_before_cache = hidden_states.copy()
            encoder_hidden_states_before_cache = encoder_hidden_states.copy()
            if not if_skip:
                hidden_states, encoder_hidden_states = self.forward_blocks_range(
                    hidden_states,
                    encoder_hidden_states,
                    block_controlnet_hidden_states,
                    temb,
                    start_idx=cache_params[0],
                    end_idx=cache_end,
                )
                delta_cache = hidden_states - hidden_states_before_cache
                delta_cache_hidden = encoder_hidden_states - encoder_hidden_states_before_cache
            else:
                hidden_states = hidden_states_before_cache + delta_cache
                encoder_hidden_states = encoder_hidden_states_before_cache + delta_cache_hidden
            
            # infer [cache_end, len(self.blocks))
            hidden_states, encoder_hidden_states = self.forward_blocks_range(
                hidden_states,
                encoder_hidden_states,
                block_controlnet_hidden_states,
                temb,
                start_idx=cache_end,
                end_idx=len(self.transformer_blocks),
            )
        return (encoder_hidden_states, hidden_states), delta_cache, delta_cache_hidden


