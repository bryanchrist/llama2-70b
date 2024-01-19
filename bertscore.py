from evaluate import load
import pandas as pd

gsm8k = pd.read_csv('data/gsm8k_questions.csv')
mathwell_all = pd.read_csv('data/mathwell_annotations.csv')
mathwell_all_good = mathwell_all[mathwell_all['good']==1]
df = pd.read_csv('data/all_models.csv')
llama = df[df['model']=='llama']
llama_good = llama[llama['good']==1]
llema = df[df['model']=='llema']
llema_good = llema[llema['good']==1]
mathwell = df[df['model']=='mathwell']
mathwell_good = mathwell[mathwell['good']==1]
mammoth = df[df['model']=='mammoth']
mammoth_good = mammoth[mammoth['good']==1]
numglue = pd.read_csv('data/numglue_questions.csv')
asdiv = pd.read_csv('data/ASDIV_clean.csv')
svamp = pd.read_json('data/svamp.json')
svamp['question'] = svamp['Body'] + " " + svamp['Question']
gsm_hard = pd.read_json('data/gsmhard.json')

def score(df1, df2, df1var, df2var):
    scores = []
    for i in range(0, len(df1)):
        for j in range(0, len(df2)):
            ref = df1.iloc[i][df1var]
            ref = str(ref)
            pred = df2.iloc[j][df2var]
            pred = str(pred)
            results = bertscore.compute(predictions=[pred], references=[ref], lang="en")
            scores.append(results)
    return scores

scores = score(gsm8k, gsm8k, 'instruction', 'instruction')
score = f"Average GSM8K overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(mathwell_all_good, mathwell_all_good, 'question', 'question')
score = f"Average MATHWELL Train overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(mathwell_all_good, gsm8k, 'question', 'instruction')
score = f"Average MATHWELL Train/GSM8K overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(mathwell, mathwell, 'question', 'question')
score = f"Average MATHWELL overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(mathwell_good, mathwell_good, 'question', 'question')
score = f"Average MATHWELL MaC overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(mathwell_good, mathwell_all_good, 'question', 'question')
score = f"Average MATHWELL MaC/MATHWELL Train overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(llama, llama, 'question', 'question')
score = f"Average llama overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(llama_good, llama_good, 'question', 'question')
score = f"Average llama MaC overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(llama,llama_good, 'question', 'question')
score = f"Average llama all generations/llama MaC overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(llema, llema, 'question', 'question')
score = f"Average llema overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(llema_good, llema_good, 'question', 'question')
score = f"Average llema MaC overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(llema,llema_good, 'question', 'question')
score = f"Average llema all generations/llema MaC overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(mammoth, mammoth, 'question', 'question')
score = f"Average mammoth overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(mammoth_good, mammoth_good, 'question', 'question')
score = f"Average mammoth MaC overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(mammoth, mammoth_good, 'question', 'question')
score = f"Average mammoth all generations/mammoth MaC overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file

scores = score(numglue, numglue, 'instruction', 'instruction')
score = f"Average numglue overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    #numglue is instruction rest are questions
    
scores = score(asdiv, asdiv, 'question', 'question')
score = f"Average asdiv overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(svamp, svamp, 'question', 'question')
score = f"Average svamp overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file
    
scores = score(gsm_hard, gsm_hard, 'question', 'question')
score = f"Average gsmhard overall BERTScore: Precision: {np.mean(scores['precision'])}, Recall: {np.mean(scores['recall'])}, F1: {np.mean(scores['f1'])}"
output_file = "bertscores.txt"  # Specify the path and filename for the output file
with open(output_file, "a") as f:  # Open the file in append mode ("a")
    f.write(score + "\n")  # Append the newly generated text to the file