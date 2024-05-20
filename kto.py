from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, setup_chat_format
import pandas as pd
import wandb
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from dotenv import load_dotenv
from collections import defaultdict
from typing import Optional, Dict, Sequence
import transformers
import os 
import numpy as np
import torch
# Load the environmental variables from the .env file
load_dotenv()

token= os.getenv('huggingface_token')
os.environ["WANDB_PROJECT"] = "mathwell8b_kto" 
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  
os.environ['TRANSFORMERS_CACHE'] = '/scratch/brc4cb/llama/cache'
wandb.login()
df = pd.read_csv('data/kto.csv')
neg_label = round(df['label'].mean()+1, 2)
pos_label = round((1-df['label'].mean())+1, 2)
dataset = load_dataset('csv', data_files="data/kto.csv")
dataset = dataset['train'].train_test_split(test_size = .05, seed = 42)

model_path = "meta-llama/Meta-Llama-3-8B"   # Specify the path to the model
adapter_path = "mathwell/llama3-8b-two_stage/checkpoint-250"  # Specify the path to the adapter weights
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right") #, use_auth_token=True,)
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True, 
       # max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        #use_auth_token=True,
        device_map = 'auto', 
    )

model_ref = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True, 
       # max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        #use_auth_token=True,
        device_map = 'auto', 
    )


# Set special tokens
DEFAULT_PAD_TOKEN = "[PAD]"
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel, model2: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    model2.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        input_embeddings = model2.get_input_embeddings().weight.data
        output_embeddings = model2.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
if tokenizer._pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model, model2 = model_ref, 
    )

print('Adding special tokens.')
tokenizer.add_special_tokens({
        "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
        "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
#                 "unk_token": tokenizer.convert_ids_to_tokens(
#                     model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
#                 ),
})

model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
model_ref = PeftModel.from_pretrained(model_ref, adapter_path, is_trainable=True)

training_args = KTOConfig(
    beta=0.1,
    desirable_weight=pos_label,
    undesirable_weight=neg_label,
    warmup_ratio = 0.1,
    num_train_epochs = 3,
    learning_rate = 1e-4, 
    per_device_train_batch_size = 8,
    output_dir="./mathwell8b-kto", 
    save_steps = 250, 
    do_eval = True,
    eval_steps = 250,
    logging_steps = 5, 
    report_to = 'wandb',
)

kto_trainer = KTOTrainer(
    model,
    model_ref,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset = dataset['test'],
    tokenizer=tokenizer,
)

kto_trainer.train()
kto_trainer.save_model("./mathwell8b-kto")