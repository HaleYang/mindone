from typing import Optional
import math
import mindspore as ms
from mindspore import ops

from mindone.diffusers.models.attention_processor import Attention as AttentionOri, AttnProcessor 
from mindone.diffusers.models.attention_processor import  JointAttnProcessor as JointAttnProcessorOri


class Attention(AttentionOri):
    def __init__(
        self, query_dim: int, 
        cross_attention_dim: int | None = None, 
        heads: int = 8, 
        dim_head: int = 64, 
        dropout: float = 0, 
        bias: bool = False, 
        upcast_attention: bool = False, 
        upcast_softmax: bool = False, 
        cross_attention_norm: str | None = None, 
        cross_attention_norm_num_groups: int = 32, 
        qk_norm: str | None = None, 
        added_kv_proj_dim: int | None = None, 
        norm_num_groups: int | None = None, 
        spatial_norm_dim: int | None = None, 
        out_bias: bool = True, scale_qk: bool = True, 
        only_cross_attention: bool = False, 
        eps: float = 0.00001, 
        rescale_output_factor: float = 1, 
        residual_connection: bool = False, 
        _from_deprecated_attn_block: bool = False, 
        processor: AttnProcessor | None = None, 
        out_dim: int = None, 
        context_pre_only=None,
        layer_idx: int = 0,
        use_todo: bool = False
    ):
        super().__init__(
            query_dim, 
            cross_attention_dim, 
            heads, 
            dim_head, 
            dropout, 
            bias, 
            upcast_attention, 
            upcast_softmax, 
            cross_attention_norm, 
            cross_attention_norm_num_groups, 
            qk_norm, 
            added_kv_proj_dim, 
            norm_num_groups, 
            spatial_norm_dim, 
            out_bias, scale_qk, 
            only_cross_attention, 
            eps, 
            rescale_output_factor, 
            residual_connection, 
            _from_deprecated_attn_block, 
            processor, 
            out_dim, 
            context_pre_only
        )

        self.layer_idx = layer_idx
        self.use_todo = use_todo
        self.use_cache = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        **cross_attention_kwargs,
    ) -> ms.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`ms.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`ms.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`ms.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `ms.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        joint_attention_kwargs = {
            "layer_idx": self.layer_idx,
            "use_downsample": self.use_todo
        } if isinstance(self.processor, JointAttnProcessor) else {}
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
            **joint_attention_kwargs,
        )


def downsample(hidden_states, merge_factor, method='nearest'):
    batch_size, _, channel = hidden_states.shape
    cur_h = int(math.sqrt(hidden_states.shape[1]))
    cur_w = cur_h
    new_h, new_w = int(cur_h / merge_factor), int(cur_w / merge_factor)
    hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, channel, cur_h, cur_w)
    merged_hidden_states = ops.interpolate(hidden_states, size=(new_h, new_w), mode=method)
    merged_hidden_states = merged_hidden_states.permute(0, 2, 3, 1).reshape(batch_size, -1, channel)
    return merged_hidden_states


@ms.jit_class
class JointAttnProcessor(JointAttnProcessorOri):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        layer_idx: int = 0,
        use_downsample: bool = False,
    ) -> ms.Tensor:
        residual = hidden_states

        batch_size, channel, height, width = (None,) * 4
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        if use_downsample and layer_idx <= 11:
            hidden_states = downsample(hidden_states, 2.2, method='bilinear')
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        # attention
        query = ops.cat([query, encoder_hidden_states_query_proj], axis=1)
        key = ops.cat([key, encoder_hidden_states_key_proj], axis=1)
        value = ops.cat([value, encoder_hidden_states_value_proj], axis=1)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        hidden_states = ops.operations.nn_ops.FlashAttentionScore(1, scale_value=attn.scale)(
            query.to(ms.float16), key.to(ms.float16), value.to(ms.float16), None, None, None, attention_mask
        )[3].to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states

