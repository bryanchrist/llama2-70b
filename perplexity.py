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

model_path = "meta-llama/Llama-2-70b-hf"  
tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    load_in_4bit=True,    # changing this to load_in_8bit=True works on smaller models
    trust_remote_code=True,
    token=token,
    device_map="auto",    # finds GPU
)

df = pd.read_csv('data/all_models.csv')
gsm8k_all = pd.read_csv('data/gsm8k_all.csv')
gsm8k_questions = pd.read_csv('data/gsm8k_questions.csv')
mathwell_all = pd.read_csv('data/mathwell_annotations_final.csv')
mathwell_all_good = mathwell_all[mathwell_all['good']==1]
llama = df[df['model']=='llama']
llama_good = llama[llama['good']==1]
llema = df[df['model']=='llema']
llema_good = llema[llema['good']==1]
mathwell = df[df['model']=='mathwell']
mathwell_good = mathwell[mathwell['good']==1]
mammoth = df[df['model']=='mammoth']
mammoth_good = mammoth[mammoth['good']==1]
sgsm_unan = pd.read_csv('data/sgsm_unannotated.csv')
sgsm = pd.concat([sgsm_unan, mathwell_all_good])

def perplexity(df):
    ppls = []
    for i in range(0, len(df)):
        text = "Question: " + str(df.iloc[i]['question']) + "\n" + "Solution:\n" + str(df.iloc[i]['solution'])
        inputs = tokenizer(text, return_tensors = "pt")
        loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
        ppl = torch.exp(loss)
        ppl = ppl.cpu().detach().numpy()
        ppls.append(ppl)
    return ppls

def perplexity_question(df):
    ppls = []
    for i in range(0, len(df)):
        text = df.iloc[i]['question']
        inputs = tokenizer(text, return_tensors = "pt")
        loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
        ppl = torch.exp(loss)
        ppl = ppl.cpu().detach().numpy()
        ppls.append(ppl)
    return ppls

def perplexity_gsm(df):
    ppls = []
    for i in range(0, len(df)):
        text = df.iloc[i]['output']
        inputs = tokenizer(text, return_tensors = "pt")
        loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
        ppl = torch.exp(loss)
        ppl = ppl.cpu().detach().numpy()
        ppls.append(ppl)
    return ppls

def perplexity_gsm_question(df):
    ppls = []
    for i in range(0, len(df)):
        text = df.iloc[i]['instruction']
        inputs = tokenizer(text, return_tensors = "pt")
        loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
        ppl = torch.exp(loss)
        ppl = ppl.cpu().detach().numpy()
        ppls.append(ppl)
    return ppls

#gsm8k_ppl = perplexity_gsm(gsm8k_all)
#gsm8k_question_ppl = perplexity_gsm_question(gsm8k_questions)
#print(f'Average GSM8K overall perplexity: {np.mean(gsm8k_ppl)} Standard Deviation: {np.std(gsm8k_ppl)}')
#print(f'Average GSM8K overall perplexity for questions only: {np.mean(gsm8k_question_ppl)} Standard Deviation: {np.std(gsm8k_question_ppl)}')

sgsm_unan_ppl = perplexity(sgsm_unan)
print(f'Average SGSM Unannotated overall perplexity: {np.mean(sgsm_unan_ppl)} Standard Deviation: {np.std(sgsm_unan_ppl)}')

sgsm_ppl = perplexity(sgsm)
print(f'Average SGSM overall perplexity: {np.mean(sgsm_ppl)} Standard Deviation: {np.std(sgsm_ppl)}')

#mathwell_all_ppl = perplexity(mathwell_all)
#mathwell_all_question_ppl = perplexity_question(mathwell_all)
#print(f'Average MATHWELL Annotated overall perplexity: {np.mean(mathwell_all_ppl)} Standard Deviation: {np.std(mathwell_all_ppl)}')
#print(f'Average MATHWELL Annotated overall perplexity for questions only: {np.mean(mathwell_all_question_ppl)} Standard Deviation: {np.std(mathwell_all_question_ppl)}')

#mathwell_all_good_ppl = perplexity(mathwell_all_good)
#mathwell_all_good_question_ppl = perplexity_question(mathwell_all_good)
#print(f'Average MATHWELL Train overall perplexity: {np.mean(mathwell_all_good_ppl)} Standard Deviation: {np.std(mathwell_all_good_ppl)}')
#print(f'Average MATHWELL Train overall perplexity for questions only: {np.mean(mathwell_all_good_question_ppl)} Standard Deviation: {np.std(mathwell_all_good_question_ppl)}')

# mathwell_ppl = perplexity(mathwell)
# #mathwell_question_ppl = perplexity_question(mathwell)
# print(f'Average MATHWELL overall perplexity: {np.mean(mathwell_ppl)} Standard Deviation: {np.std(mathwell_ppl)}')
# #print(f'Average MATHWELL overall perplexity for questions only: {np.mean(mathwell_question_ppl)} Standard Deviation: {np.std(mathwell_question_ppl)}')

# mathwell_good_ppl = perplexity(mathwell_good)
# #mathwell_good_question_ppl = perplexity_question(mathwell_good)
# print(f'Average MATHWELL Good overall perplexity: {np.mean(mathwell_good_ppl)} Standard Deviation: {np.std(mathwell_good_ppl)}')
# #print(f'Average MATHWELL Good overall perplexity for questions only: {np.mean(mathwell_good_question_ppl)} Standard Deviation: {np.std(mathwell_good_question_ppl)}')

# llama_ppl = perplexity(llama)
# #llama_question_ppl = perplexity_question(llama)
# print(f'Average Llama overall perplexity: {np.mean(llama_ppl)} Standard Deviation: {np.std(llama_ppl)}')
# #print(f'Average Llama overall perplexity for questions only: {np.mean(llama_question_ppl)} Standard Deviation: {np.std(llama_question_ppl)}')

# llama_good_ppl = perplexity(llama_good)
# #llama_good_question_ppl = perplexity_question(llama_good)
# print(f'Average Llama Good overall perplexity: {np.mean(llama_good_ppl)} Standard Deviation: {np.std(llama_good_ppl)}')
# #print(f'Average Llama Good overall perplexity for questions only: {np.mean(llama_good_question_ppl)} Standard Deviation: {np.std(llama_good_question_ppl)}')

# llema_ppl = perplexity(llema)
# #llema_question_ppl = perplexity_question(llema)
# print(f'Average Llemma overall perplexity: {np.mean(llema_ppl)} Standard Deviation: {np.std(llema_ppl)}')
# #print(f'Average Llemma overall perplexity for questions only: {np.mean(llema_question_ppl)} Standard Deviation: {np.std(llema_question_ppl)}')

# llema_good_ppl = perplexity(llema_good)
# #llema_good_question_ppl = perplexity_question(llema_good)
# print(f'Average Llemma Good overall perplexity: {np.mean(llema_good_ppl)} Standard Deviation: {np.std(llema_good_ppl)}')
# #print(f'Average Llemma Good overall perplexity for questions only: {np.mean(llema_good_question_ppl)} Standard Deviation: {np.std(llema_good_question_ppl)}')

# mammoth_ppl = perplexity(mammoth)
# #mammoth_question_ppl = perplexity_question(mammoth)
# print(f'Average Mammoth overall perplexity: {np.mean(mammoth_ppl)} Standard Deviation: {np.std(mammoth_ppl)}')
# #print(f'Average Mammoth overall perplexity for questions only: {np.mean(mammoth_question_ppl)} Standard Deviation: {np.std(mammoth_question_ppl)}')

# mammoth_good_ppl = perplexity(mammoth_good)
# #mammoth_good_question_ppl = perplexity_question(mammoth_good)
# print(f'Average Mammoth Good overall perplexity: {np.mean(mammoth_good_ppl)} Standard Deviation: {np.std(mammoth_good_ppl)}')
# #print(f'Average Mammoth Good overall perplexity for questions only: {np.mean(mammoth_good_question_ppl)} Standard Deviation: {np.std(mammoth_good_question_ppl)}')
