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

appropriateness = df[df['appropriateness']>=0]

prompt = appropriateness.query("question=='During a week of intense battles, the Power Rangers defeated 15 evil monsters on Monday, twice as many on Tuesday as on Monday, and 10 fewer on Wednesday than on Tuesday. How many monsters did they defeat in total from Monday to Wednesday?'")
prompt1 = {'question' : prompt['question'].iloc[0], 
#'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'appropriateness': prompt['appropriateness'].iloc[0],
'explanation': "This question is written at an appropriate mathematical difficulty for K-8 students, contains appropriate language, is not strange or unrealistic, is free from grammar errors and typos, and requires mathematical operations to arrive at the final correct answer."}

prompt = appropriateness.query("question=='Serena Williams has won a certain number of tennis matches this season. If she wins 8 more matches, she will double the number she has won so far. How many matches has Serena won this season?'")
prompt2 = {'question' : prompt['question'].iloc[0], 
#'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'appropriateness': prompt['appropriateness'].iloc[0],
'explanation': "This question does not require any mathematical operation to solve and is therefore not appropriate. The question tells you Serena's number of wins will double if she wins 8 more matches, which means she had to have won 8 matches so far."}

prompt = appropriateness.query("question=='A cheerleading team has 20 members. They want to evenly divide into 4 squads for a competition. How many members will be on each squad?'")
prompt3 = {'question' : prompt['question'].iloc[0], 
#'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'appropriateness': prompt['appropriateness'].iloc[0],
'explanation': "This question is written at an appropriate mathematical difficulty for K-8 students, contains appropriate language, is not strange or unrealistic, is free from grammar errors and typos, and requires mathematical operations to arrive at the final correct answer."}

prompt = appropriateness.query("question=='In a Pokémon battle, Pikachu has a 60% chance of winning each round. If Pikachu and Ash battle for 5 rounds, what is the probability that Pikachu wins all 5 rounds?'")
prompt4 = {'question' : prompt['question'].iloc[0], 
#'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'appropriateness': prompt['appropriateness'].iloc[0],
'explanation': "This question is too hard for a middle school student. It assumes a student knows about the probability of independent events, which is typically not covered until high school or an introductory statistics college course."}

prompt = appropriateness.query("question=='Hello Kitty makes 18 bracelets in 4 hours. How many bracelets per hour does she make?'")
prompt5 = {'question' : prompt['question'].iloc[0], 
#'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'appropriateness': prompt['appropriateness'].iloc[0],
'explanation': "This question is written at an appropriate mathematical difficulty for K-8 students, contains appropriate language, is not strange or unrealistic, is free from grammar errors and typos, and requires mathematical operations to arrive at the final correct answer."}

prompt = appropriateness.query("question=='Batman caught a baddie with his trademark punch. Each punch knocks out 7 baddies. If Batman has thrown 60 punches, how many baddies has he knocked out?'")
prompt6 = {'question' : prompt['question'].iloc[0], 
#'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'appropriateness': prompt['appropriateness'].iloc[0],
'explanation': "While this question is comical, it is not appropriate for a K-8 student because it involves physically harming another person."}

prompt = appropriateness.query("question=='Taylor Swift has 11 Grammys, 29 AMAs, 12 CMAs, 8 ACMs and 35 BMAs. How many awards has she won in total?'")
prompt7 = {'question' : prompt['question'].iloc[0], 
#'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'appropriateness': prompt['appropriateness'].iloc[0],
'explanation': "This question is written at an appropriate mathematical difficulty for K-8 students, contains appropriate language, is not strange or unrealistic, is free from grammar errors and typos, and requires mathematical operations to arrive at the final correct answer."}

prompt = appropriateness.query("question=='The soccer team has 32 players. Each player has 2 legs. How many legs does the team have?'")
prompt8 = {'question' : prompt['question'].iloc[0], 
#'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'appropriateness': prompt['appropriateness'].iloc[0],
'explanation': "While this question is solvable, it is not appropriate because it is strange to ask how many legs a soccer team has."}

prompt = appropriateness.query("question=='A cat has 100 kittens. 20 of them are calico, 30 are tabby, and the rest are siamese. How many kittens are siamese?'")
prompt9 = {'question' : prompt['question'].iloc[0], 
#'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'appropriateness': prompt['appropriateness'].iloc[0],
'explanation': "This question is not based in reality, as it is not possible for one cat to birth 100 kittens, nor is it possible for them to be different breeds."}

prompt = appropriateness.query("model=='mathwell' & answer=='3.0' & appropriateness==0.0")
prompt10 = {'question' : prompt['question'].iloc[0], 
#'solution': prompt['solution'].iloc[0],
'answer': prompt['answer'].iloc[0], 
'appropriateness': prompt['appropriateness'].iloc[0],
'explanation': "This question is inappropriate to give to a student because it does not require any mathematical operations to solve. It directly defines the number of forwards on the team."}

prompts = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10]
for i in prompts:
    appropriateness = appropriateness[appropriateness['question']!=i['question']]

import json
responses = []
json_responses = []
agreement = []
for i in range(0, len(appropriateness)):
    query = {'question': appropriateness.iloc[i]['question'], 
    #'solution': appropriateness.iloc[i]['solution'], 
    'answer': appropriateness.iloc[i]['answer']}
    ground_truth = appropriateness.iloc[i]['appropriateness']         
    appropriateness_prompt = f"""You are a teacher tasked with evaluating math word problems for K-8 students. You must evaluate each question for educational appropriateness, meaning whether it is appropriate for a student in a K-8 classroom setting. There are four primary reasons why a question should be flagged as educationally inappropriate: being strange or unrealistic (some exaggeration is fine for fictional settings, but the question should not contain factually inaccurate statements such as a cat having 7 toes), being too difficult for a K-8 student (as assessed by containing any mathematical operation beyond addition, subtraction, multiplication, division, fractions or decimals or having an answer that would be difficult for a student to arrive at without a calculator), containing inappropriate content for a classroom setting (such as describing harming someone else), or having grammatical errors or typos. A final reason why a question would be labeled as educationally inappropriate is that it does not require any mathematical operations to solve and, therefore, is a reading comprehension question rather than a math word problem. 

You will be given a question, solution and answer in JSON format and be asked to complete the complete the JSON object with a 1 or 0 to denote that a question is appropriate or not appropriate, respectively, along with a brief explanation for your answer.   
    
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
            "content": f"{appropriateness_prompt}",
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
        annotation = my_json['appropriateness']
        if annotation == ground_truth:
            agreement.append(1)
        else:
            agreement.append(0)
    
    except:
        message = message.split("appropriateness")[1]
        message = message.split(": ")[1]
        annotation = message.split(",")[0]
        if annotation == ground_truth:
            agreement.append(1)
        else:
            agreement.append(0)
    
    if i%50==0:
        output_file = "gpt4_appropriateness_annotations.txt"  # Specify the path and filename for the output file
        with open(output_file, "a") as f:  # Open the file in append mode ("a")
            f.write(f"Average appropriateness agreement at question {i}: {np.mean(agreement)}\n")  # Append the newly generated text to the file

gpt4_annotations = pd.DataFrame.from_dict(responses)
gpt4_annotations.to_csv('data/gpt4_appropriateness_annotations.csv')
gpt4_annotations_json = pd.DataFrame.from_dict(json_responses)
gpt4_annotations_json.to_csv('data/gpt4_appropriateness_annotations_json.csv')