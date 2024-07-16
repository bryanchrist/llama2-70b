from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()
api_key = os.getenv('gpt4_api_key')
azure_endpoint = os.getenv('gpt4_azure_endpoint')

# gets the API Key from environment variable AZURE_OPENAI_API_KEY
client = AzureOpenAI(
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
    api_version="2023-07-01-preview",
    api_key=api_key,
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint=azure_endpoint,
)

import pandas as pd
import numpy as np
df = pd.read_csv('data/annotations.csv')

accuracy = df[df['accuracy']>=0]

prompt = accuracy.query("question=='Jenny has 15 cats. She buys 2 new cat towers, each can hold 4 cats. If she places as many cats as possible on the cat towers, how many cats are not on the towers?'")
prompt1 = {'question' : prompt['question'].iloc[0], 
'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'accuracy': prompt['accuracy'].iloc[0],
'explanation': "The solution for this question correctly identifies the necessary variables, calculates the number of cats the towers can hold by multiplying the capacity of each tower by the number of towers, and then subtracts the capacity of the towers from the number of cats Jenny has to arrive at the correct number of cats that are not on the tower."}

prompt = accuracy.query("model=='gpt4' & answer=='108'")
prompt2 = {'question' : prompt['question'].iloc[0], 
'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'accuracy': prompt['accuracy'].iloc[0],
'explanation': "The solution adds the original number of kittens to the number of new kittens, which is incorrect because the original kittens are now considered adults based on the text of the question."}

prompt = accuracy.query("question=='Black Panther has 1500 vibranium shards. He needs to divide them equally amongst his 5 closest allies. How many vibranium shards will each ally receive?'")
prompt3 = {'question' : prompt['question'].iloc[0], 
'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'accuracy': prompt['accuracy'].iloc[0],
'explanation': "The solution correctly defines the necessary variables and divides the number of vibranium shards by the number of allies to arrive at the correct answer."}

prompt = accuracy.query("question=='A dog shelter has 40 dogs. If each dog needs 2 cups of food per day, and a bag of food contains 40 cups, how many bags of food will the shelter need for a 30-day period?'")
prompt4 = {'question' : prompt['question'].iloc[0], 
'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'accuracy': prompt['accuracy'].iloc[0],
'explanation': "The solution adds an additional bag of food to the total assuming that the answer is a decimal and, therefore, that the shelter would need to buy another full bag since a partial bag is not possible. However, since the answer is whole number, this additional bag leads to outputting an incorrect result."}

prompt = accuracy.query("question=='Naruto eats 4 bananas a day. How many bananas does he eat in a month if there are 30 days in a month?'")
prompt5 = {'question' : prompt['question'].iloc[0], 
'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'accuracy': prompt['accuracy'].iloc[0],
'explanation': "The solution arrives at the correct answer by multiplying the number of bananas Naruto eats per day by the number of days in a month."}

prompt = accuracy.query("question=='12 cats eat 400 pounds of cat food every Saturday morning. 15 cats eat 500 pounds of cat food on Sunday morning. How many pounds of cat food are eaten in total?'")
prompt6 = {'question' : prompt['question'].iloc[0], 
'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'accuracy': prompt['accuracy'].iloc[0],
'explanation': "The solution multiplies the number of cats by the amount of cat food eaten, rather than adding the two sums of cat food together."}

prompt = accuracy.query("question=='Harry Potter and his friends have just finished their exams and are looking forward to a well-deserved break. They decide to go on a camping trip together. They have 120 Galleons between them. They spend 30 Galleons on food in the morning and 20 Galleons on food in the afternoon. They have 20 Galleons left. How many Galleons did they spend in the evening?'")
prompt7 = {'question' : prompt['question'].iloc[0], 
'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'accuracy': prompt['accuracy'].iloc[0],
'explanation': "The solution correctly defines the necessary variables and then correctly determines the amount of Galleons spent in the evening by subtracting the amount spent in the morning, the amount spent in the afternoon, and the amount remaining."}

prompt = accuracy.query("question=='Captain Marvel has 100 friends on Facebook. She has 40 more friends than the average number of friends her friends have. How many friends does the average friend of Captain Marvel have?'")
prompt8 = {'question' : prompt['question'].iloc[0], 
'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'accuracy': prompt['accuracy'].iloc[0],
'explanation': "The solution calculates an average rather than subtract the average number of friends Captain Marvel's friends have from her number of friends."}

prompt = accuracy.query("question=='A basketball team scored 120 points in a game. The team scored 30 points in the first quarter, 35 points in the second quarter, 20 points in the third quarter, and 35 points in the fourth quarter. How many points did the team score in the second half of the game?'")
prompt9 = {'question' : prompt['question'].iloc[0], 
'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'accuracy': prompt['accuracy'].iloc[0],
'explanation': "The solution correctly defines the variables, but it incorrectly adds the points scored in the second quarter to the total for the points scored in the second half."}

prompt = accuracy.query("question=='A Minecraft player has 100000 blocks. 20000 of the blocks are dirt, 30000 of the blocks are stone, 20000 of the blocks are wood, and 30000 of the blocks are diamond. How many of the blocks are not dirt, stone, wood, or diamond?'")
prompt10 = {'question' : prompt['question'].iloc[0], 
'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'accuracy': prompt['accuracy'].iloc[0],
'explanation': "The solution correctly defines the necessary variables, but does not subtract the number of diamond blocks from the total number of blocks."}

prompts = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10]
for i in prompts:
    accuracy = accuracy[accuracy['question']!=i['question']]

import json
responses = []
json_responses = []
agreement = []
for i in range(0, len(accuracy)):
    query = {'question': accuracy.iloc[i]['question'], 
    'solution': accuracy.iloc[i]['solution'], 
    'answer': accuracy.iloc[i]['answer']}
    ground_truth = accuracy.iloc[i]['accuracy']         
    accuracy_prompt = f"""You are a teacher tasked with evaluating word problems for K-8 students. You must evaluate each question's solution for accuracy, which means the Python function solution provided arrives at the correct answer for the question and does not engage in any rounding unless the question specifically asks for it. You will be given a question, solution and answer in JSON format and be asked to complete the complete the JSON object with a 1.0 or 0.0 to denote that a question's solution is accurate or not accurate, respectively, along with a brief explanation for your answer.  
    
Here are some examples: 
{prompt1}
    
{prompt2}
    
{prompt3}
    
{prompt4}
    
{prompt5}
    
{prompt6}
    
{prompt7}

{prompt8}

{prompt9}

{prompt10}

Now evaluate this question: 
{query}
"""
    completion = client.chat.completions.create(
    model="gpt-4-turbo",  
    messages=[
        {
            "role": "user",
            "content": f"{accuracy_prompt}",
        },
    ],
    )
    message = completion.choices[0].message.content
    
    try:
        message = message.split("```json")[1]
        message = message.split("```")[0]
    
    except:
        message = message

    responses.append(message)
    
    try: 
        my_json = json.loads(message)
        json_responses.append(my_json)
        annotation = my_json['accuracy']
        if annotation == ground_truth:
            agreement.append(1)
        else:
            agreement.append(0)
    
    except:
        try:
            message = message.split("accuracy")[1]
            message = message.split(": ")[1]
            annotation = message.split(",")[0]
            if annotation == ground_truth:
                agreement.append(1)
            else:
                agreement.append(0)
        except:
            pass
    
    if i%50==0:
        output_file = "gpt4_accuracy_annotations.txt"  # Specify the path and filename for the output file
        with open(output_file, "a") as f:  # Open the file in append mode ("a")
            f.write(f"Average accuracy agreement at question {i}: {np.mean(agreement)}\n")  # Append the newly generated text to the file

gpt4_annotations = pd.DataFrame.from_dict(responses)
gpt4_annotations.to_csv('data/gpt4_accuracy_annotations.csv')
gpt4_annotations_json = pd.DataFrame.from_dict(json_responses)
gpt4_annotations_json.to_csv('data/gpt4_accuracy_annotations_json.csv')