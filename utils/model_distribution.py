import torch 
import yaml
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

def model_distribution(model,fsdp_group,ddp_group,rank):
    for i in range(len(model.model.layers)):
        model.model.layers[i]=FSDP(
            model.model.layers[i],
            process_group=fsdp_group,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=rank,
        )
    model.model.embed_tokens=FSDP(
            model.model.embed_tokens,
            process_group=fsdp_group,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=rank,
        )
    
    fsdp_model = FSDP(
        model,
        process_group=fsdp_group,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=rank,
    )

    ddp_fsdp_model = DDP(
        fsdp_model,
        process_group=ddp_group,  # 关键：指定DDP使用的进程组
        device_ids=[rank],
    )
    return ddp_fsdp_model