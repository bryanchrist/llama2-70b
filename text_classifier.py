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
torch.cuda.init()
print(torch.cuda.is_available())
import os
os.environ['TRANSFORMERS_CACHE'] = '/project/SDS/research/christ_research/Llama 2/llama2-7b/cache'
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
    PeftModel,
    TaskType
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from dotenv import load_dotenv
import os
torch.backends.cuda.matmul.allow_tf32 = True

# Load the environmental variables from the .env file
load_dotenv()

token = os.getenv('text_classifier_token')

from huggingface_hub import login
login(token = token)

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

# Test dataset splitting worked
#print(train_test_valid_dataset['train'][1])

#Set up tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=True)
tokenizer.add_special_tokens({"pad_token":"[PAD]"})

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_auth_token=True)
# tokenizer.add_special_tokens({"pad_token":"[PAD]"})

#Preprocess and collate data
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_train_test_valid_dataset = train_test_valid_dataset.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Prepare evaluation function
import evaluate
accuracy = evaluate.load("accuracy")

import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

#Training
id2label = {0: "NOT SOLVABLE", 1: "SOLVABLE"}
label2id = {"NOT SOLVABLE": 0, "SOLVABLE": 1}

#Import model
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b-hf", num_labels=2, id2label=id2label, label2id=label2id,
        use_auth_token=True,  
      # max_memory=max_memory,
      # torch_dtype=torch.bfloat16, 
      device_map = 'auto',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'))

# model = AutoModelForSequenceClassification.from_pretrained(
#     "bert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id,
#         use_auth_token=True,  
#       # max_memory=max_memory,
#       # torch_dtype=torch.bfloat16, 
#         quantization_config=BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type='nf4'))
        
#NEWLY ADDED
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)      

model = get_peft_model(model, peft_config)

#END NEWLY ADDED
training_args = TrainingArguments(
    output_dir="text_classifier_llama",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
   # auto_find_batch_size = True,
    num_train_epochs=8,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True, 
    bf16=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_test_valid_dataset["train"],
    eval_dataset=tokenized_train_test_valid_dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.predict(test_dataset = tokenized_train_test_valid_dataset["test"])
trainer.evaluate(eval_dataset = tokenized_train_test_valid_dataset["test"])