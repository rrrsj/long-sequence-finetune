from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils.load_data import load_gsm8k
import torch 

class MyDataset(Dataset):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.tokenizer=AutoTokenizer.from_pretrained(self.config['model']['tokenzier_path'])
        self.input_data=[]
        self.output_data=[]
        self.ans_mask=[]
        self.max_length=self.config['training']['max_length']
        self.pad_token_id=self.tokenizer.pad_token_id
        self.init_dataset(self.config['data']['data_path'],self.config['data']['data_name'])
    def init_dataset(self,data_path,dataset_name):
        if dataset_name=='gsm8k':
            self.input_data,self.output_data,self.ans_mask=load_gsm8k(data_path,self.tokenizer)


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self,index):
        input_index=self.input_data[index]
        output_index=self.output_data[index]
        mask_index=self.ans_mask[index]
        input_index=input_index+[self.pad_token_id for i in range(self.max_length-len(input_index))]
        output_index=output_index+[self.pad_token_id for i in range(self.max_length-len(output_index))]
        mask_index=mask_index+[0 for i in range(self.max_length-len(mask_index))]
        return torch.tensor(input_index),torch.tensor(output_index),torch.tensor(mask_index)
        
        