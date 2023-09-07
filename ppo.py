from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd

import torch
import os
os.environ['TRANSFORMERS_CACHE'] = '/project/SDS/research/christ_research/Llama 2/llama2-70b-hf/cache'
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from dotenv import load_dotenv
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
from datasets import load_dataset
from peft import PeftModel, PeftConfig, LoraConfig, TaskType

# trl: Transformer Reinforcement Learning library
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler

import torch
import evaluate

import numpy as np
import pandas as pd

# tqdm library makes the loops show a smart progress meter.
from tqdm import tqdm
tqdm.pandas()

# Load the environmental variables from the .env file
load_dotenv()

token = os.getenv('huggingface_token')

from huggingface_hub import login
login(token = token)

#Load in base model
model_path = "meta-llama/Llama-2-70b-hf"   # Specify the path to the model
adapter_path = "output/checkpoint-2000"  # Specify the path to the adapter weights
tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)

model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True, 
       # max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        use_auth_token=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'))
        
peft_model = PeftModel.from_pretrained(model, adapter_path)

#Set up PPO Model
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model,                                                               
                                                               torch_dtype=torch.bfloat16,
                                                               is_trainable=True)
                                                               
ref_model = create_reference_model(ppo_model)

from datasets import load_dataset, DatasetDict

# Load the dataset
dataset = load_dataset('csv', data_files="app/feedback.csv")

# Get the total number of examples in the dataset
total_examples = len(dataset['train'])

# Calculate the sizes of the training, test, and validation sets
train_size = int(0.8 * total_examples)
test_size = int(0.1 * total_examples)
valid_size = total_examples - train_size - test_size

# Manually split the dataset into training, test, and validation sets
train_dataset = dataset['train'].shuffle(seed=42).select(range(train_size))
test_dataset = dataset['train'].shuffle(seed=42).select(range(train_size, train_size + test_size))
valid_dataset = dataset['train'].shuffle(seed=42).select(range(train_size + test_size, total_examples))

# Create a DatasetDict to hold the splits
train_test_valid_dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
    'valid': valid_dataset
})

#Preprocess and collate data
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_train_test_valid_dataset = train_test_valid_dataset.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "NOT SOLVABLE", 1: "SOLVABLE"}
label2id = {"NOT SOLVABLE": 0, "SOLVABLE": 1}

#Set up classifier
solvability_model_name = "text_classifier/checkpoint-3744"
solvability_model =AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id,
        use_auth_token=True,  
      # max_memory=max_memory,
      # torch_dtype=torch.bfloat16, 
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'))
            
solvability_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_auth_token=True)
tokenizer.add_special_tokens({"pad_token":"[PAD]"})
solvability_model = PeftModel.from_pretrained(solvability_model, solvability_model_name)
solvability_index = 1

#Set up collator
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

#Set up PPO trainer    
learning_rate=1.41e-5
max_ppo_epochs=8
mini_batch_size=4
batch_size=16

config = PPOConfig(
    model_name=model_path,    
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size
)

ppo_trainer = PPOTrainer(config=config, 
                         model=ppo_model, 
                         ref_model=ref_model, 
                         tokenizer=tokenizer, 
                         dataset=train_test_valid_dataset["train"], 
                         data_collator=collator)
                         
#Set up and implement training
generation_kwargs = {
    "min_length": 5,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True
}

reward_kwargs = {
    "top_k": None, # Return all scores.
    "function_to_apply": "none", # You want the raw logits without softmax.
    "batch_size": 16
}

max_ppo_steps = 5000

prompt = "Write one math word problem and Python code with a commented out step-by-step solution to solve the word problem."
formatted_prompt = (f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response: Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"
            f" Solution: def solution():\n    #Roger started with 5 tennis balls\n    tennis_balls = 5\n    #2 cans of 3 tennis balls each is\n    bought_balls = 2 * 3    \n    #tennis balls. The answer is\n    result = tennis_balls + bought_balls\n    return result"
            f"\nBelow is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response: Question: The bakers at the Beverly Hills Bakery baked 200 loaves of bread on Monday morning. "
            f"They sold 93 loaves in the morning and 39 loaves in the afternoon. A grocery store returned 6 unsold loaves. How many loaves of bread did they have left?"
            f" Solution: def solution():\n    #The bakers started with 200 loaves\n    loaves_baked = 200\n    #They sold 93 in the morning and 39 in the afternoon\n    loaves_sold_morning=93\n    loaves_sold_afternoon=39\n    "
            f"#The grocery store returned 6 loaves\n    loaves_returned = 6\n    #The answer is\n    result = loaves_baked - loaves_sold_morning - loaves_sold_afternoon + loaves_returned\n    return result"
            f"\nBelow is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response: Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
            f" Solution: def solution():\n    #Olivia started with $23\n    money_initial = 23\n    #She bought 5 bagels\n    bagels = 5\n    #Each bagel cost $3\n    bagel_cost = 3\n    #5 bagels at $3 a bagel cost\n    money_spent = bagels * bagel cost\n"
            f"    #The answer is\n    result = money_initial - money_spent\n    return result"
            f"\nBelow is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response: Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday? Solution: def solution()\n"
            f"    #Michael started with 58 golf balls\n    golf_balls_initial = 58\n    #He lost 23 on Tuesday\n    golf_balls_lost_tuesday = 23\n    #He lost two more on Wednesday\n    golf_balls_lost_wednesday = 2\n    #The answer is\n    "
            f"result = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n    return result"
            f"\nBelow is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response: Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room? Solution: def solution():\n"
            f"    #There were initially 9 computers\n    computers_initial = 9\n    #They installed 5 more each day\n    computers_per_day = 5\n    #There are 4 days between Monday and Thursday\n    num_days = 4\n    #There were\n    "
            f"computers_added = computers_per_day * num_days\n    #computers added. The answer is\n    result = computers_initial + computers_added\n    return result"
            f"\nBelow is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:")
            
for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # Break when you reach max_steps.
    if step >= max_ppo_steps:
        break   

    # formatted_prompt = [formatted_prompt] * len(batch["label"])

    # for i, label_value in enumerate(batch["label"]):
    #     batch["label"][i] = {"input_ids": formatted_prompt[i], "label": label_value}

    # prompt_tensors = batch['label']["input_ids"]

    # Get response from LLM.
    question_tensors = []
    
    # for prompt_tensor in prompt_tensors:
    #     max_new_tokens = 800       
            
    #     generation_kwargs["max_new_tokens"] = max_new_tokens
    #     inputs = tokenizer.encode(prompt_tensor, return_tensors="pt")
    #     attention_mask = torch.ones_like(inputs)
    #     inputs = inputs.to('cuda')
    #     print(inputs.shape)
    #     question = ppo_trainer.generate(inputs, attention_mask=attention_mask, **generation_kwargs)
        
    #     question_tensors.append(question.squeeze()[-max_new_tokens:])

    for i in batch:
        max_new_tokens = 800       
            
        generation_kwargs["max_new_tokens"] = max_new_tokens
        generation_kwargs['do_sample'] = True
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        attention_mask = torch.ones_like(inputs)
        inputs = inputs.to('cuda')
        print(inputs.shape)
        question = ppo_trainer.generate(inputs, attention_mask=attention_mask, **generation_kwargs)
        
        question_tensors.append(question.squeeze()[-max_new_tokens:])
        
    # This needs to be called "response".
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in question_tensors]

    # Compute reward outputs.
    #query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]    
    solvability_input_ids = solvability_tokenizer(batch['response'], return_tensors="pt").input_ids

    logits = solvability_model(solvability_input_ids).logits

    # You use the solvability item because this is the score for the positive solvability class.
    reward_tensors = [torch.tensor(reward[solvability_index]["score"]) for reward in rewards]    

    # Run PPO step.
    stats = ppo_trainer.step(prompt_tensors, question_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, batch, reward_tensors)
    
    print(f'objective/kl: {stats["objective/kl"]}')
    print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
    print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')

# # Define prompt_tensors using the formatted_prompt
# prompt_tensors = tokenizer.encode(formatted_prompt, return_tensors="pt")
# attention_mask = torch.ones_like(prompt_tensors)
# prompt_tensors = prompt_tensors.to('cuda')
# prompt_tensors = prompt_tensors.unsqueeze(0) 
# print(prompt_tensors.shape)

# for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
#     # Break when you reach max_steps.
#     if step >= max_ppo_steps:
#         break   

#     # Get response from LLM.
#     question_tensors = []

#     max_new_tokens = 800
#     generation_kwargs["max_new_tokens"] = max_new_tokens

#     question = ppo_trainer.generate(prompt_tensors, attention_mask=attention_mask, **generation_kwargs)
#     question_tensors.append(question.squeeze()[-max_new_tokens:])

#     # This needs to be called "response".
#     batch["response"] = [tokenizer.decode(r.squeeze()) for r in question_tensors]

#     # Compute reward outputs.
#     solvability_input_ids = solvability_tokenizer(batch['response'], return_tensors="pt").input_ids
#     logits = solvability_model(solvability_input_ids).logits

#     # You use the solvability item because this is the score for the positive solvability class.
#     reward_tensors = [torch.tensor(reward[solvability_index]["score"]) for reward in rewards]    

#     # Run PPO step.
#     stats = ppo_trainer.step(prompt_tensors, question_tensors, reward_tensors)
#     ppo_trainer.log_stats(stats, batch, reward_tensors)
    
#     print(f'objective/kl: {stats["objective/kl"]}')
#     print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
#     print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
#     print('-'.join('' for x in range(100)))

#     print('-'.join('' for x in range(100)))