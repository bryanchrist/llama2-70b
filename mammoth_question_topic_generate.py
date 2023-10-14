import os
import sys
import builtins
#os.environ['TRANSFORMERS_CACHE'] = '/project/SDS/research/christ_research/Llama 2/mammoth/cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/brc4cb/mammoth/cache'
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
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
import random

import torch
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
#from adapter-transformers import AdapterType, AdapterConfig, load_adapter

# Set the environment variable
os.environ["HF_REMOTES_OFFLINE"] = "1"
from dotenv import load_dotenv
# Load the environmental variables from the .env file
load_dotenv()

token = os.getenv('huggingface_token')
if token:
    print('token loaded')
    
token = os.environ['huggingface_token']
#Uncomment if needed
from huggingface_hub import login
login(token=token)

# Redirect stdin to /dev/null
sys.stdin = open(os.devnull)

model_path = "TIGER-Lab/MAmmoTH-70B"   # Specify the path to the model
adapter_path = "mammoth_question_adapter/checkpoint-3750"  # Specify the path to the adapter weights
tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)

# Patch the built-in input function to return 'y' automatically
def mock_input(prompt=None):
    return 'y'

# Patch the input function to use the mock_input function
builtins.input = mock_input

try:
    # Attempt to load the model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True, 
       # max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        use_auth_token=True,
        config=AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    )
except EOFError:
    # If an EOFError occurs, provide the expected input ('y')
    pass

# Restore stdin
sys.stdin = sys.__stdin__

# Load the adapter weights
model = PeftModel.from_pretrained(model, adapter_path)

import pandas as pd

df = pd.read_json('mammoth_question_train.json')
df2 = pd.read_json('ASDiv_instruct.json')
grades = ['1', '2', '3', '4', '5', '6']
for i in range(0,100):
    
    prompt = f"Write a grade school math word problem."
    final_prompt = "Write a grade school math word problem about Superman."
    questions = []
    for i in range(0, 3):
        temp_df = df.query(f"instruction=='{prompt}'")
        question = temp_df['output'].iloc[random.randint(0,len(temp_df)-1)]
        questions.append(question)
    formatted_prompt = []
    for i in range(0,3):
        formatted_prompt.append((f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{prompt}\n\n### Response: {questions[i]}"))
    formatted_prompt.append(f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{final_prompt}\n\n### Response: ")
    formatted_prompt = "\n".join(formatted_prompt)
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)
    inputs = inputs.to('cuda')
    output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 400, do_sample = True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Split the generated text by the prompt to extract the newly generated part
    generated_text_parts = generated_text.split(prompt)
    newly_generated_text = generated_text_parts[-1].strip()
    
    output_file = "mammoth_question_topic_generate.txt"  # Specify the path and filename for the output file
    with open(output_file, "a") as f:  # Open the file in append mode ("a")
        f.write(f"Prompting Approach: few shot. Generated Text: " + newly_generated_text + "\n")  # Append the newly generated text to the file
        
    # prompt = f"Write a grade school math word problem."
    # inputs = tokenizer.encode(prompt, return_tensors="pt")
    # attention_mask = torch.ones_like(inputs)
    # inputs = inputs.to('cuda')
    # output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 400, do_sample = True)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # # Split the generated text by the prompt to extract the newly generated part
    # generated_text_parts = generated_text.split(prompt)
    # newly_generated_text = generated_text_parts[-1].strip()
    
    # with open(output_file, "a") as f:  # Open the file in append mode ("a")
    #     f.write(f"Prompting Approach: zero shot. Generated Text: " + newly_generated_text + "\n")  # Append the newly generated text to the file
        
    # grade = random.choice(grades)
    # prompt = f"Write a grade {grade} math word problem."
    # questions = []
    # for i in range(0, 7):
    #     temp_df = df2.query(f"instruction=='{prompt}'")
    #     question = temp_df['output'].iloc[random.randint(0,len(temp_df)-1)]
    #     questions.append(question)
    # formatted_prompt = []
    # for i in range(0,7):
    #     formatted_prompt.append((f"Below is an instruction that describes a task. "
    #             f"Write a response that appropriately completes the request.\n\n"
    #             f"### Instruction:\n{prompt}\n\n### Response: {questions[i]}"))
    # formatted_prompt.append(f"Below is an instruction that describes a task. "
    #             f"Write a response that appropriately completes the request.\n\n"
    #             f"### Instruction:\nWrite a grade {grade} math word problem.\n\n### Response: ")
    # formatted_prompt = "\n".join(formatted_prompt)
    # inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    # attention_mask = torch.ones_like(inputs)
    # inputs = inputs.to('cuda')
    # output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 400, do_sample = True)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # # Split the generated text by the prompt to extract the newly generated part
    # generated_text_parts = generated_text.split(prompt)
    # newly_generated_text = generated_text_parts[-1].strip()
    
    # with open(output_file, "a") as f:  # Open the file in append mode ("a")
    #     f.write(f"Prompting Approach: few shot. Target Grade Level: {grade}. Generated Text: " + newly_generated_text + "\n")  # Append the newly generated text to the file