import os
import sys
import builtins
#os.environ['TRANSFORMERS_CACHE'] = '/project/SDS/research/christ_research/Llama 2/llama2-70b/cache'
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

token = os.getenv('huggingface_token')
if token:
    print('token loaded')
    
token = os.environ['huggingface_token']
#Uncomment if needed
from huggingface_hub import login
login(token=token)

# Redirect stdin to /dev/null
sys.stdin = open(os.devnull)

model_path = "meta-llama/Llama-2-70b-hf"   # Specify the path to the model
adapter_path = "mathwell/llama_QA_adapter_no_embed/checkpoint-1250"  # Specify the path to the adapter weights
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
        torch_dtype=torch.float16,
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
import random

df = pd.read_csv('data/evaluation_annotations.csv')
df = df[df['good']==1]
df['output'] = "Question: " + df['question'] + "\n" + "Solution:\n" + df['solution']

addition = df.query("addition==1 and total_ops==1")
subtraction = df.query("subtraction==1 and total_ops==1")
multiplication = df.query("multiplication==1 and total_ops==1")
division = df.query("division==1 and total_ops==1")
fractions = df.query("fractions==1 and total_ops==1")
decimals = df.query("decimals==1 and total_ops==1")
multi_ops = df.query("total_ops>1")

topics = ['Superman', "Batman", "Wonder Woman", "Barbie", "Power Rangers", "basketball", "soccer", "football", "volleyball", 'field hockey',\
'Fortnite', 'Spiderman', "Iron Man", "Captain America", "Captain Marvel", "Thor, the God of Thunder", "Ninja Turtles", "Black Panther", "Taylor Swift", "swimming",\
"Pokémon", "Super Mario", "Naruto", "unicorns", "Hello Kitty", "Minecraft", "lacrosse", "cheer leading", "LeBron James", "Steph Curry", "Patrick Mahomes",\
"Serena Williams", "dogs", "cats", "dinosaurs", "Harry Potter", "cars", "planes", "trains", "pizza", "cookies", "ice cream", 'candy']
def prompt(df, model, tokenizer, n_qs, operation, operator, topics):
    responses = []
    while len(responses)<n_qs:
        topic = random.choice(topics)
        final_prompt = f"Write a grade school math {operation} word problem about {topic} and Python function with a commented out step-by-step solution to solve the word problem. The question you write should only require {operation} to solve, meaning the solution should rely only on use of the {operator} operator."
        prompt = f"Write a grade school math {operation} word problem and Python function with a commented out step-by-step solution to solve the word problem. The question you write should only require {operation} to solve, meaning the solution should rely only on use of the {operator} operator."
        #final_prompt = f"Write a grade school math {operation} word problem about {topic} and Python function with a commented out step-by-step solution to solve the word problem."
        #prompt = f"Write a grade school math {operation} word problem and Python function with a commented out step-by-step solution to solve the word problem."
        questions = []
        while len(questions)<8:
            question = df['output'].iloc[random.randint(0,len(df)-1)]
            if question not in questions:
                questions.append(question)
        formatted_prompt = []
        for i in range(0,8):
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
        output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 200, do_sample = True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Split the generated text by the prompt to extract the newly generated part
        generated_text_parts = generated_text.split(final_prompt)
        newly_generated_text = generated_text_parts[-1].strip()
        if "\nBel" in newly_generated_text:
            newly_generated_text = newly_generated_text.split("\nBel")[0]
        if "Solution:" in newly_generated_text:
            if "return" in newly_generated_text:
                responses.append(newly_generated_text)
    return responses
        # output_file = "mathwell8b_kto_questions_few.txt"  # Specify the path and filename for the output file
        # with open(output_file, "a") as f:  # Open the file in append mode ("a")
        #     f.write(f"Topic: {topic} " + newly_generated_text + "\n")  # Append the newly generated text to the file
        
qs = prompt(addition, model, tokenizer, 100, 'addition', "+", topics=topics)
addition_df = pd.DataFrame(qs)
addition_df['operation'] = 'addition'

q_len = len(qs)
add = 0
success = []
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if " - " not in solution and " * " not in solution and "/" not in solution and "." not in solution and "+" in solution:
            add+=1
            success.append(1)
        else:
            success.append(0)
    except:
        q_len-=1
        
addition_df['success']= success
        
output_file = "controllability.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a") 
    f.write(f"Percentage with just addition: {add/q_len}" + "\n")  # Append the newly generated text to the file

print(f"Percentage with just addition: {add/q_len}")

q_len = len(qs)
add = 0
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if " + " in solution:
            add+=1
    except:
        q_len-=1
print(f"Percentage with addition overall: {add/q_len}")
output_file = "controllability.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a") 
    f.write(f"Percentage with addition overall: {add/q_len}" + "\n") 
    
qs = prompt(subtraction, model, tokenizer, 100, 'subtraction', "-", topics=topics)
subtraction_df = pd.DataFrame(qs)
subtraction_df['operation'] = 'subtraction'

q_len = len(qs)
sub = 0
success = []
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if " + " not in solution and " * " not in solution and "/" not in solution and "." not in solution and " - " in solution:
            sub+=1
            success.append(1)
        else:
            success.append(0)
    except:
        q_len-=1
        
subtraction_df['success']= success     

print(f"Percentage with subtraction overall: {sub/q_len}")
output_file = "controllability.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a") 
    f.write(f"Percentage with subtraction only: {sub/q_len}" + "\n") 
    
q_len = len(qs)
sub = 0
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if " - " in solution:
            sub+=1
    except:
        q_len-=1
print(f"Percentage with subtraction overall: {sub/q_len}")
output_file = "controllability.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a") 
    f.write(f"Percentage with subtraction overall: {sub/q_len}" + "\n")
    
    
qs = prompt(multiplication, model, tokenizer, 100, 'multiplication', "*", topics=topics)
multiplication_df = pd.DataFrame(qs)
multiplication_df['operation'] = 'multiplication'

q_len = len(qs)
mult = 0
success = []
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if " + " not in solution and " - " not in solution and "/" not in solution and "." not in solution and "*" in solution:
            mult+=1
            success.append(1)
        else: 
            success.append(0)
    except:
        q_len-=1

multiplication_df['success']= success    

print(f"Percentage with multiplication only: {mult/q_len}")
output_file = "controllability.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a") 
    f.write(f"Percentage with multiplication only: {mult/q_len}" + "\n")

q_len = len(qs)
mult = 0
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if "*" in solution:
            mult+=1
    except:
        q_len-=1
print(f"Percentage with multiplication overall: {mult/q_len}")
output_file = "controllability.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a") 
    f.write(f"Percentage with multiplication overall: {mult/q_len}" + "\n")
    
qs = prompt(division, model, tokenizer, 100, 'division', "/", topics=topics)
division_df = pd.DataFrame(qs)
division_df['operation'] = 'division'

q_len = len(qs)
div = 0
success = []
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if " + " not in solution and " - " not in solution and " * " not in solution and "." not in solution and "/" in solution:
            div+=1
            success.append(1)
        else: 
            success.append(0)
    except:
        q_len-=1
        
division_df['success']= success 

print(f"Percentage with division only: {div/q_len}")
output_file = "controllability.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a") 
    f.write(f"Percentage with division only: {div/q_len}" + "\n")

q_len = len(qs)
div = 0
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if "/" in solution:
            div+=1
    except:
        q_len-=1
print(f"Percentage with division overall: {div/q_len}")
output_file = "controllability.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a") 
    f.write(f"Percentage with division overall: {div/q_len}" + "\n")
    
all_ops = pd.concat([addition_df, subtraction_df, division_df, multiplication_df])
all_ops.to_csv('controllability_samples.csv')