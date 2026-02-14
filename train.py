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
import sys
from tqdm import tqdm
from utils.model_distribution import model_distribution
from utils.set_seed import set_all_seeds

dist.init_process_group(backend="nccl")
config_path='./config.yaml'
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
set_all_seeds(config['training']['seed'])

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device_map=config['training']['device_map']
sys.stdout = open(f'logs/rank_{rank}.log', 'w')
sys.stderr = sys.stdout

all_process=device_map[0]*device_map[1]
assert all_process==world_size
assert config['training']['max_length']%device_map[1]==0

torch.cuda.set_device(local_rank)
mp_number=rank//(device_map[1])
my_ddp_group,my_fsdp_group=create_gpu_group(rank,world_size,device_map[0],device_map[1])
dataset=MyDataset(config)

sampler = DistributedSampler(
        dataset,
        num_replicas=device_map[0],  
        rank=mp_number,               
        shuffle=True,
        drop_last=True,
    )

dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )
model=MyQwen3_4B(config,device_map,rank,my_fsdp_group,my_ddp_group,config['training']['max_length'])
model.monkey_patch()
model=model_distribution(model,my_fsdp_group,my_ddp_group,rank)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rating'])

fsdp_rank=(int(os.environ["RANK"])%device_map[1])
length_start_id=(config['training']['max_length']//device_map[1])*(fsdp_rank)
length_end_id=(config['training']['max_length']//device_map[1])*(fsdp_rank+1)
for i in range(config['training']['epoch']):
    for j in tqdm(dataloader):
        model_input,model_output,loss_index=j
        model_input=model_input[:,length_start_id:length_end_id]
        position_ids=torch.tensor([length_start_id+i for i in range(length_end_id-length_start_id)]).reshape(1,-1).expand(model_input.shape[0],-1)
        output=model(model_input,position)
        break






