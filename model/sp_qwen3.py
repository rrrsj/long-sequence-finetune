import transformers
import torch
import torch.nn as nn
import types
from transformers import AutoModelForCausalLM,AutoConfig
from layer.qwen3_attention_forward import new_forward
from layer.sp_layer import SPAttentionReduce,SPAttentionGather




def parse_torch_dtype(dtype_str):
    if dtype_str == "auto":
        return "auto"
    if dtype_str.startswith("torch."):
        return getattr(torch, dtype_str.split("torch.")[1])
    raise ValueError(f"不支持的 dtype: {dtype_str}")



class MyQwen3_4B(nn.Module):
    def __init__(self,config,device_map,rank,fsdp_group,ddp_group,max_length):
        super().__init__()
        
        self.config=config
        self.device_map=device_map
        self.rank=rank
        self.fsdp_group=fsdp_group
        self.ddp_group=ddp_group
        self.max_length=max_length
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
        Qwen3Attention.forward = new_forward

        if self.config['training']['load_type']=='lazy':
            self.model,self.lm_head=self.init_model_empty(self.config['model']['model_path'],parse_torch_dtype(self.config['model']['dtype']))
        elif self.config['training']['load_type']=='full':
            self.model=AutoModelForCausalLM.from_pretrained(self.config['model']['model_path'],dtype=parse_torch_dtype(self.config['model']['dtype']),device_map='cpu')
            self.lm_head=self.model.lm_head
            self.model=self.model.model

        
    def init_model_empty(self,model_path,dtype):
        config = AutoConfig.from_pretrained(model_path)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
            lm_head=model.lm_head
            model=model.model
        for name, param in model.named_parameters():
            assert param.device == torch.device("meta")
        return model,lm_head

    def forward(self,x,position):
        inputs=self.model(input_ids=x,position_ids=position)['last_hidden_state']
        outputs=self.lm_head(inputs)
        return outputs


    def monkey_patch(self):
        for i in range(len(self.model.layers)):
            self.model.layers[i].self_attn.device_map=self.device_map
            self.model.layers[i].self_attn.max_length=self.max_length
            self.model.layers[i].self_attn.fsdp_group=self.fsdp_group
            self.model.layers[i].self_attn.ddp_group=self.ddp_group
            self.model.layers[i].self_attn.reduce=SPAttentionReduce()
            self.model.layers[i].self_attn.gather=SPAttentionGather()