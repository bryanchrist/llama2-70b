import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/brc4cb/llama/cache'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd
import random

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", device_map="auto", torch_dtype=torch.bfloat16)

df = pd.read_csv('data/sgsm.csv') # Load SGSM dataset for few-shot prompting
df = df[df['subset']=="sgsm_train"] # Subset SGSM to verified training subset
df = df.sample(frac = 1, random_state = 42)
for i in range(0, len(df)):
    try:
        answer = df.iloc[i]['answer']
        answer = float(answer)
        df.iloc[i]['answer'] = answer
    except:
        df = df.drop([i])
        
exec_code = []
solved = []
for i in range(0, len(df)-6):
    # Format the prompt
    prompts = []
    questions = []
    final_question = df.iloc[i]['question']
    answer = df.iloc[i]['answer']
    final_prompt = f"""Instruct: {final_question} Let's write a Python program.\nOutput:"""
    
    for j in range(len(df)-5, len(df)):
        question = df['question'].iloc[j]
        solution = df['solution'].iloc[j]
        prompt = f"""Instruct: {question} Let's write a Python program.\nOutput:\n{solution}"""
        if prompt not in prompts:
            prompts.append(prompt)

    prompts.append(final_prompt)
    formatted_prompt = "\n".join(prompts)

    #Query the model 
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to('cuda')
    output = model.generate(inputs, max_new_tokens = 100)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Split the generated text by the prompt to extract the newly generated part
    generated_text_parts = generated_text.split(final_prompt)
    solution_text = generated_text_parts[-1].strip()
    if "<|end|>" in solution_text:
            solution_text = solution_text.split("<|end|>")[0] # Split up a generation that contains more than one question
    if "<|user|>" in solution_text:
            solution_text = solution_text.split("<|user|>")[0] # Split up a generation that contains more than one question
    try:
        exec(solution_text)
        model_answer = solution()
        model_answer = float(model_answer)
        if model_answer != answer:
            solved.append(0)
            exec_code.append(1)
                

        if model_answer == answer:
            solved.append(1)
            exec_code.append(1)

    except:
        solved.append(0)
        exec_code.append(0)
        pass
print(np.mean(solved)*100)
print(np.mean(exec_code)*100)
        
print(f"First attempt solve rate % for phi 1.5: {np.mean(solved)*100}")
print(f"% exec code for phi 1.5: {np.mean(exec_code)*100}")

output_file = "phi_test.txt"  # Specify the path and filename for the output file

with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(f"""-----------First attempt solve rate % for phi 1.5: {np.mean(solved)*100}""" + "\n" + f"% exec code for phi 1.5: {np.mean(exec_code)*100}" + "\n\n ----")  # Append the newly generated text to the file
    
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

exec_code = []
solved = []
for i in range(0, len(df)-6):
    # Format the prompt
    prompts = []
    questions = []
    final_question = df.iloc[i]['question']
    answer = df.iloc[i]['answer']
    final_prompt = f"""Instruct: {final_question} Let's write a Python program.\nOutput:"""
    
    for j in range(len(df)-5, len(df)):
        question = df['question'].iloc[j]
        solution = df['solution'].iloc[j]
        prompt = f"""Instruct: {question} Let's write a Python program.\nOutput:\n{solution}"""
        if prompt not in prompts:
            prompts.append(prompt)

    prompts.append(final_prompt)
    formatted_prompt = "\n".join(prompts)

    #Query the model 
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to('cuda')
    output = model.generate(inputs, max_new_tokens = 100)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Split the generated text by the prompt to extract the newly generated part
    generated_text_parts = generated_text.split(final_prompt)
    solution_text = generated_text_parts[-1].strip()
    if "<|end|>" in solution_text:
            solution_text = solution_text.split("<|end|>")[0] # Split up a generation that contains more than one question
    if "<|user|>" in solution_text:
            solution_text = solution_text.split("<|user|>")[0] # Split up a generation that contains more than one question
    try:
        exec(solution_text)
        model_answer = solution()
        model_answer = float(model_answer)
        if model_answer != answer:
            solved.append(0)
            exec_code.append(1)
                

        if model_answer == answer:
            solved.append(1)
            exec_code.append(1)

    except:
        solved.append(0)
        exec_code.append(0)
        pass
print(np.mean(solved)*100)
print(np.mean(exec_code)*100)
        
print(f"First attempt solve rate % for llama3 8b instruct: {np.mean(solved)*100}")
print(f"% exec code for llama3 8b instruct: {np.mean(exec_code)*100}")

output_file = "phi_test.txt"  # Specify the path and filename for the output file

with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(f"""-----------First attempt solve rate % for llama3 8b instruct: {np.mean(solved)*100}""" + "\n" + f"% exec code for llama3 8b instruct: {np.mean(exec_code)*100}" + "\n\n ----")  # Append the newly generated text to the file
    
model_id = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

exec_code = []
solved = []
for i in range(0, len(df)-6):
    # Format the prompt
    prompts = []
    questions = []
    final_question = df.iloc[i]['question']
    answer = df.iloc[i]['answer']
    final_prompt = f"""Instruct: {final_question} Let's write a Python program.\nOutput:"""
    
    for j in range(len(df)-5, len(df)):
        question = df['question'].iloc[j]
        solution = df['solution'].iloc[j]
        prompt = f"""Instruct: {question} Let's write a Python program.\nOutput:\n{solution}"""
        if prompt not in prompts:
            prompts.append(prompt)

    prompts.append(final_prompt)
    formatted_prompt = "\n".join(prompts)

    #Query the model 
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to('cuda')
    output = model.generate(inputs, max_new_tokens = 100)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Split the generated text by the prompt to extract the newly generated part
    generated_text_parts = generated_text.split(final_prompt)
    solution_text = generated_text_parts[-1].strip()
    if "<|end|>" in solution_text:
            solution_text = solution_text.split("<|end|>")[0] # Split up a generation that contains more than one question
    if "<|user|>" in solution_text:
            solution_text = solution_text.split("<|user|>")[0] # Split up a generation that contains more than one question
    try:
        exec(solution_text)
        model_answer = solution()
        model_answer = float(model_answer)
        if model_answer != answer:
            solved.append(0)
            exec_code.append(1)
                

        if model_answer == answer:
            solved.append(1)
            exec_code.append(1)

    except:
        solved.append(0)
        exec_code.append(0)
        pass
print(np.mean(solved)*100)
print(np.mean(exec_code)*100)
        
print(f"First attempt solve rate % for llama3 8b: {np.mean(solved)*100}")
print(f"% exec code for llama3 8b: {np.mean(exec_code)*100}")

output_file = "phi_test.txt"  # Specify the path and filename for the output file

with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(f"""-----------First attempt solve rate % for llama3 8b: {np.mean(solved)*100}""" + "\n" + f"% exec code for llama3 8b: {np.mean(exec_code)*100}" + "\n\n ----")  # Append the newly generated text to the file