import os
import sys
import builtins
#os.environ['TRANSFORMERS_CACHE'] = '/project/SDS/research/christ_research/Llama 2/mammoth/cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/brc4cb/llama/cache'
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

token= os.getenv('huggingface_token')

from huggingface_hub import login
login(token = token)

# Redirect stdin to /dev/null
sys.stdin = open(os.devnull)

model_path = "meta-llama/Meta-Llama-3-8B"   # Specify the path to the model
# adapter_path = "output/checkpoint-2000"  # Specify the path to the adapter weights
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
# model = PeftModel.from_pretrained(model, adapter_path)
import pandas as pd
import random
df = pd.read_csv('data/mathwell.csv')
topics = ['Superman', "Batman", "Wonder Woman", "Barbie", "Power Rangers", "basketball", "soccer", "football", "volleyball", 'field hockey',\
'Fortnite', 'Spiderman', "Iron Man", "Captain America", "Captain Marvel", "Thor, the God of Thunder", "Ninja Turtles", "Black Panther", "Taylor Swift", "swimming",\
"Pokémon", "Super Mario", "Naruto", "unicorns", "Hello Kitty", "Minecraft", "lacrosse", "cheer leading", "LeBron James", "Steph Curry", "Patrick Mahomes",\
"Serena Williams", "dogs", "cats", "dinosaurs", "Harry Potter", "cars", "planes", "trains", "pizza", "cookies", "ice cream", 'candy']
for i in range(0,5000):
    topic = random.choice(topics)
    final_prompt = f"Write a grade school math word problem about {topic} and Python function with a commented out step-by-step solution to solve the word problem."
    prompt = "Write a grade school math word problem and Python function with a commented out step-by-step solution to solve the word problem."
    questions = []
    for i in range(0, 5):
        question = df['output'].iloc[random.randint(0,len(df)-1)]
        questions.append(question)
    formatted_prompt = []
    for i in range(0,5):
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
    output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 250, do_sample = True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Split the generated text by the prompt to extract the newly generated part
    generated_text_parts = generated_text.split(final_prompt)
    newly_generated_text = generated_text_parts[-1].strip()
    if "\nBel" in newly_generated_text:
        newly_generated_text = newly_generated_text.split("\nBel")[0]
    output_file = "llama3_8b_questions.txt"  # Specify the path and filename for the output file
    with open(output_file, "a") as f:  # Open the file in append mode ("a")
        f.write(f"Topic: {topic} " + newly_generated_text + "\n")  # Append the newly generated text to the file
        