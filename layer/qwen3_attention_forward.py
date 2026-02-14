import torch
import torch.nn as nn
import os
from layer.sp_layer import SPAttentionGather,SPAttentionReduce
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    eager_attention_forward,
)

def new_forward(
        self,
        hidden_states ,
        position_embeddings ,
        attention_mask,
        past_key_values ,
        cache_position,
        **kwargs,
    ):
    input_shape = hidden_states.shape[:-1]

    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    query_states=query_states.transpose(1,2)
    key_states=key_states.transpose(1,2)
    value_states=value_states.transpose(1,2)
    assert query_states.shape[-2]%self.device_map[1]==0
    assert key_states.shape[-2]%self.device_map[1]==0
    assert value_states.shape[-2]%self.device_map[1]==0
    assert self.max_length%self.device_map[1]==0

    query_states=self.gather(query_states,self.fsdp_group,self.device_map,query_states.shape[-2],self.max_length)
    key_states=self.gather(key_states,self.fsdp_group,self.device_map,key_states.shape[-2],self.max_length)
    value_states=self.gather(value_states,self.fsdp_group,self.device_map,value_states.shape[-2],self.max_length)
    query_states=query_states.transpose(1,2)
    key_states=key_states.transpose(1,2)
    value_states=value_states.transpose(1,2)
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output=self.reduce(attn_output,self.fsdp_group,self.device_map,value_states.shape[-2],self.max_length)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights

