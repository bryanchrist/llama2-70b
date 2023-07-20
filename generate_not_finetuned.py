import os
import sys
import builtins
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

# Redirect stdin to /dev/null
sys.stdin = open(os.devnull)

model_path = "huggyllama/llama-65b"  # Specify the path to the model
adapter_path = "output/guanaco-65b/checkpoint-2000"  # Specify the path to the adapter weights
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Patch the built-in input function to return 'y' automatically
def mock_input(prompt=None):
    return 'y'

# Patch the input function to use the mock_input function
builtins.input = mock_input

try:
    # Attempt to load the model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True, 
       # max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
        config=AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    )
except EOFError:
    # If an EOFError occurs, provide the expected input ('y')
    pass

# Restore stdin
sys.stdin = sys.__stdin__

tokenizer = AutoTokenizer.from_pretrained(model_path)

for i in range(0,5):
    prompt = "Write a math word problem and Python code to solve the word problem."
    formatted_prompt = (f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response: Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? Solution: def solution():\n    cars_initial = 3\n    cars_arrived = 2\n"
            f"    result = cars_initial + cars_arrived\n    return result"
            f"\nBelow is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response: Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? Solution: def solution():\n    leah_chocolates = 32\n    sister_chocolates = 42\n"
            f"    total_chocolates = leah_chocolates + sister_chocolates\n    chocolates_eaten = 35\n    result = total_chocolates - chocolates_eaten\n    return result"
            f"\nBelow is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response: ")
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)
    inputs = inputs.to('cuda')
    output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 400, temperature = .8, do_sample = True)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(generated_text)
    output_file = "output_no_tune.txt"  # Specify the path and filename for the output file
    with open(output_file, "a") as f:  # Open the file in append mode ("a")
        f.write(generated_text + "\n")  # Append the generated text to the file
    
    
    # prompt = "Write a math word problem and step-by-step solution to solve the word problem."
    # formatted_prompt = (f"Below is an instruction that describes a task. "
    #         f"Write a response that appropriately completes the request.\n\n"
    #         f"### Instruction:\n{prompt}\n\n### Response:")
    # inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    # attention_mask = torch.ones_like(inputs)
    # inputs = inputs.to('cuda')
    # output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 400, temperature = .8)
    
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # print(generated_text)
    # output_file = "output.txt"  # Specify the path and filename for the output file
    # with open(output_file, "a") as f:  # Open the file in append mode ("a")
    #     f.write(generated_text + "\n")  # Append the generated text to the file

    # prompt = "Write a math word problem and Python code with a commented out step-by-step solution to solve the word problem."
    # formatted_prompt = (f"Below is an instruction that describes a task. "
    #         f"Write a response that appropriately completes the request.\n\n"
    #         f"### Instruction:\n{prompt}\n\n### Response:")
    # inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    # attention_mask = torch.ones_like(inputs)
    # inputs = inputs.to('cuda')
    # output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 400, temperature = .8)
    
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # print(generated_text)
    # output_file = "output.txt"  # Specify the path and filename for the output file
    # with open(output_file, "a") as f:  # Open the file in append mode ("a")
    #     f.write(generated_text + "\n")  # Append the generated text to the file
        
    prompt = "Write a math word problem and Python code with a commented out step-by-step solution to solve the word problem."
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
            f" Solution: def solution():\n   #Olivia started with $23\n    money_initial = 23\n    #She bought 5 bagels\n    bagels = 5\n    #Each bagel cost $3\n    bagel_cost = 3\n    #5 bagels at $3 a bagel cost\n    money_spent = bagels * bagel cost\n"
            f"    #The answer is\n    result = money_initial - money_spent\n    return result"
            f"\nBelow is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:")
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)
    inputs = inputs.to('cuda')
    output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 400, do_sample = True)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(generated_text)
    output_file = "output_no_tune.txt"  # Specify the path and filename for the output file
    with open(output_file, "a") as f:  # Open the file in append mode ("a")
        f.write(generated_text + "\n")  # Append the generated text to the file
        
        
print("Generated text appended to", output_file)
