import json
import pandas as pd
from transformers import AutoTokenizer
import os

def load_gsm8k(data_path,tokenizer):
    data_input=[]
    data_output=[]
    data_mask=[]
    pad_token_id=tokenizer.pad_token_id
    files = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    for file in files:
        df = pd.read_parquet(file)
        for i in range(len(df['question'])):
            question=df['question'][i]
            ans=df['answer'][i]
            question_index=tokenizer(question)['input_ids']
            ans_index=tokenizer(ans)['input_ids']
            input_index=question_index+ans_index
            output_index=question_index[1:]+ans_index+[pad_token_id]
            loss_mask=[0 for i in range(len(question_index[1:]))]+[1 for i in range(len(ans_index)+1)]
            assert len(input_index)==len(output_index)
            assert len(input_index)==len(loss_mask)
            data_input.append(input_index)
            data_output.append(output_index)
            data_mask.append(loss_mask)
    return data_input,data_output,data_mask






if __name__=='__main__':
    tokenizer=AutoTokenizer.from_pretrained("./checkpoint/old_checkpoint/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/eb25fbe4f35f7147763bc24445679d1c00588d89")
    load_gsm8k('./data/datasets--openai--gsm8k/snapshots/cc7b047b6e5bb11b4f1af84efc572db110a51b3c/main/',tokenizer)
    

