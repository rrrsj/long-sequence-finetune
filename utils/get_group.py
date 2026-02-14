import torch
import torch.distributed as dist

def create_gpu_group(rank,world_size,ddp_group,fsdp_group):
    my_ddp_group=None
    my_fsdp_group=None

    for i in range(ddp_group):
        now_group=[j for j in range(i*fsdp_group,(i+1)*fsdp_group)]
        group = dist.new_group(now_group)
        if rank in now_group:
            my_fsdp_group=group

    for i in range(fsdp_group):
        now_group=[i+j*fsdp_group for j in range(ddp_group)]
        group = dist.new_group(now_group)
        if rank in now_group:
            my_ddp_group=group
    return my_ddp_group,my_fsdp_group
