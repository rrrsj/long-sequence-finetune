import torch 
import yaml
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from model.sp_qwen3 import MyQwen3_4B
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed as dist
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from utils.get_group import create_gpu_group
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from dataset.mydataset import MyDataset
from torch.nn.parallel import DistributedDataParallel as DDP

def model_distribution(model,fsdp_group,ddp_group,rank,device_map):
    auto_wrap_policy=ModuleWrapPolicy({nn.Linear})
    mesh=DeviceMesh("cuda",torch.arrange(dist.get_world_size()).view(device_map),mesh_dim_names=("ddp,fsdp"))
    ddp_fsdp_model=FSDP(model,sharding_strategy=ShardingStrategy.HYBRID_SHARD,device_mesh=mesh,auto_wrap_policy=auto_wrap_policy,device_id=rank)
    return ddp_fsdp_model
