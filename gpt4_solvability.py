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
solvability = df

prompt1 = solvability.query("question=='In a volleyball match, Team A scored twice as many points as Team B. If Team B scored 15 points, how many points in total were scored in the match?'")
first_prompt = {'question' : prompt1['question'].iloc[0], 
#'solution': prompt1['solution'].iloc[0],
#'answer': prompt1['answer'].iloc[0], 
'solvability': prompt1['solvability'].iloc[0],
'explanation': "This question arrives at one correct answer and contains all the information needed to solve it"}

prompt1 = solvability.query("question=='In a Fortnite Battle Royale match, there are 100 players. If one player can eliminate 4 opponents, how many players in the game can achieve such an elimination count, assuming no other players are eliminated?'")
prompt2 = {'question' : prompt1['question'].iloc[0], 
#'solution': prompt1['solution'].iloc[0],
#'answer': prompt1['answer'].iloc[0], 
'solvability': prompt1['solvability'].iloc[0],
'explanation': "It is not entirely clear what this question is asking. If it defined how many players should be left in the game after eliminations, it would be possible to determine how many players can be eliminated but, without defining that, it is not clear what the student needs to do to solve this problem."}

prompt1 = solvability.query("question=='During a basketball game, the Lakers made a total of 84 points. LeBron James scored 35 points, and Anthony Davis scored 20 points. How many points were scored by the rest of the Lakers team?'")
prompt3 = {'question' : prompt1['question'].iloc[0], 
#'solution': prompt1['solution'].iloc[0],
#'answer': prompt1['answer'].iloc[0], 
'solvability': prompt1['solvability'].iloc[0],
'explanation': "This question arrives at one correct answer and contains all the information needed to solve it"}

prompt1 = solvability.query("question=='The Ninja Turtles went to the pizza parlor and ordered 8 pizzas. Each pizza had 8 slices. If they each ate 2 slices, how many slices of pizza were left over?'")
prompt4 = {'question' : prompt1['question'].iloc[0], 
#solution': prompt1['solution'].iloc[0],
#'answer': prompt1['answer'].iloc[0], 
'solvability': prompt1['solvability'].iloc[0],
'explanation': "This question does not define the number of Ninja Turtles. While many students might know there are typically 4 main Ninja Turtles, not every student would know this and all math questions should still define the key variables you need to answer them."}

prompt1 = solvability.query("question=='Steph Curry makes 2133 free throws for the year. He hits 342 more free throws than he misses. How many free throws does Steph Curry miss?'")
prompt5 = {'question' : prompt1['question'].iloc[0], 
#'solution': prompt1['solution'].iloc[0],
#'answer': prompt1['answer'].iloc[0], 
'solvability': prompt1['solvability'].iloc[0],
'explanation': "This question arrives at one correct answer and contains all the information needed to solve it"}

prompt1 = solvability.query("question=='There are 5175 Pokémon available to battle in Pokémon Sword and Shield. There are 310 ground-type Pokémon and 182 water-type Pokémon. There are 610 Pokémon that are not ground nor water type. How many Pokémon are ground-type or water-type?'")
prompt6 = {'question' : prompt1['question'].iloc[0], 
#'solution': prompt1['solution'].iloc[0],
#'answer': prompt1['answer'].iloc[0], 
'solvability': prompt1['solvability'].iloc[0],
'explanation': "The question contains conflicting information in that it defines the number of ground and water type Pokémon twice."}
prompt1 = solvability.query("question=='In Fortnite, the player has 5 health points. When the player is hit by an enemy, they lose 2 health points. How many health points does the player have left?'")

prompt7 = {'question' : prompt1['question'].iloc[0], 
#'solution': prompt1['solution'].iloc[0],
#'answer': prompt1['answer'].iloc[0], 
'solvability': prompt1['solvability'].iloc[0],
'explanation': "This question defines how many health points a player loses when they get hit by an enemy, but it does not define how many times the player got hit, so it is impossible to determine how many health points they have left. "}

prompts = [first_prompt, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7]
for i in prompts:
    solvability = solvability.query(f"question!='{i['question']}'")

import json
responses = []
json_responses = []
agreement = []
for i in range(0, len(solvability)):
    query = {'question': solvability.iloc[i]['question']} 
    #'solution': solvability.iloc[i]['solution'], 
    #'answer': solvability.iloc[i]['answer']}
    ground_truth = solvability.iloc[i]['solvability']         
    solvability_prompt = f"""You are a teacher tasked with evaluating word problems for K-8 students. You must evaluate each question for solvability, which means it has one correct answer, contains the information necessary to solve the problem, and does not arrive at a negative answer when it is not possible to have a negative number (for example, having negative items left over after giving some to someone). You will be given a question, solution and answer in JSON format and be asked to complete the complete the JSON object with a 1 or 0 to denote that a question is solvable or not solvable, respectively, along with a brief explanation for your answer. 
    
Here are some examples: 
{first_prompt}
    
{prompt2}
    
{prompt3}
    
{prompt4}
    
{prompt5}
    
{prompt6}
    
{prompt7}
    
Now evaluate this question: 
{query}
"""
    completion = client.chat.completions.create(
    model="gpt-4-turbo",  
    messages=[
        {
            "role": "user",
            "content": f"{solvability_prompt}",
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
        annotation = my_json['solvability']
        if annotation == ground_truth:
            agreement.append(1)
        else:
            agreement.append(0)
    
    except:
        try:
            message = message.split("solvability")[1]
            message = message.split(": ")[1]
            annotation = message.split(",")[0]
            if annotation == ground_truth:
                agreement.append(1)
            else:
                agreement.append(0)
        except: 
            pass
    
    if i%50==0:
        output_file = "gpt4_solvability_annotations.txt"  # Specify the path and filename for the output file
        with open(output_file, "a") as f:  # Open the file in append mode ("a")
            f.write(f"Average solvability agreement at question {i}: {np.mean(agreement)}\n")  # Append the newly generated text to the file
        gpt4_annotations = pd.DataFrame.from_dict(responses)
        gpt4_annotations.to_csv('data/gpt4_solvability_annotations.csv')
        gpt4_annotations_json = pd.DataFrame.from_dict(json_responses)
        gpt4_annotations_json.to_csv('data/gpt4_solvability_annotations_json.csv')
        
gpt4_annotations = pd.DataFrame.from_dict(responses)
gpt4_annotations.to_csv('data/gpt4_solvability_annotations.csv')
gpt4_annotations_json = pd.DataFrame.from_dict(json_responses)
gpt4_annotations_json.to_csv('data/gpt4_solvability_annotations_json.csv')