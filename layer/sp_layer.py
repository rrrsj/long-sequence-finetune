import torch 
import torch.nn as nn
from torch.autograd import Function
import os
import torch.distributed as dist



class sp_attention_gather(Function):
    @staticmethod
    def forward(ctx,input_data,sp_group,device_map,num_head,max_length):
        ctx.group = sp_group
        ctx.world_size=int(os.environ["WORLD_SIZE"])
        ctx.fsdp_rank=(int(os.environ["RANK"])%device_map[1])
        ctx.sp_group=sp_group
        ctx.device_map=device_map
        ctx.max_length=max_length
        if ctx.world_size==1:
            return input_data
        else:
            head_pre_gpu=(input_data.shape[2]//device_map[1])
            input_list = [input_data[:,:,head_pre_gpu*i:head_pre_gpu*(i+1),:].contiguous() for i in range(device_map[1])]#batch,seq,head,dim
            gather_data = [torch.empty_like(input_list[0]) for i in range(device_map[1])]
            dist.all_to_all(gather_data, input_list, group=sp_group)
            all_seq_data=torch.concat(gather_data,dim=1)
            return all_seq_data.contiguous()

    @staticmethod
    def backward(ctx,grad):
        if ctx.world_size==1:
            return grad, None, None, None, None
        else:
            length_pre_gpu=ctx.max_length//ctx.device_map[1]
            input_list=[grad[:,length_pre_gpu*i:length_pre_gpu*(i+1),:,:].contiguous() for i in range(ctx.device_map[1])]
            gather_data = [torch.empty_like(input_list[0]) for i in range(ctx.device_map[1])]
            dist.all_to_all(gather_data, input_list, group=ctx.sp_group)
            all_grad=torch.concat(gather_data,dim=2)
            return all_grad.contiguous(), None, None, None, None

    
class SPAttentionGather(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_data,sp_group, device_map, num_head, max_length):
        return sp_attention_gather.apply(
            input_data,
            sp_group,
            device_map,
            num_head,
            max_length,
        )



class sp_attention_reduce(Function):
    @staticmethod
    def forward(ctx,input_data,sp_group,device_map,num_head,max_length):
        ctx.group = sp_group
        ctx.world_size=int(os.environ["WORLD_SIZE"])
        ctx.fsdp_rank=(int(os.environ["RANK"])%device_map[1])
        ctx.sp_group=sp_group
        ctx.device_map=device_map
        ctx.max_length=max_length
        if ctx.world_size==1:
            return input_data, None, None, None, None
        else:
            length_pre_gpu=ctx.max_length//device_map[1]
            input_list=[input_data[:,length_pre_gpu*i:length_pre_gpu*(i+1),:,:].contiguous() for i in range(device_map[1])]
            gather_data = [torch.empty_like(input_list[0]) for i in range(device_map[1])]
            dist.all_to_all(gather_data, input_list, group=sp_group)
            all_seq_data=torch.concat(gather_data,dim=2)
            return all_seq_data

    @staticmethod
    def backward(ctx,grad):
        if ctx.world_size==1:
            return grad
        else:
            num_head=grad.shape[2]
            head_pre_gpu=num_head//ctx.device_map[1]
            input_list=[grad[:,:,head_pre_gpu*i:head_pre_gpu*(i+1),:].contiguous() for i in range(ctx.device_map[1])]
            gather_data = [torch.empty_like(input_list[0]) for i in range(ctx.device_map[1])]
            dist.all_to_all(gather_data, input_list, group=ctx.sp_group)
            all_grad=torch.concat(gather_data,dim=1)
            return all_grad.contiguous(), None, None, None, None


class SPAttentionReduce(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_data,sp_group, device_map, num_head, max_length):
        return sp_attention_reduce.apply(
            input_data,
            sp_group,
            device_map,
            num_head,
            max_length,
        )

