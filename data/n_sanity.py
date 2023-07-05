import jiwer 
import numpy as np
from tqdm import tqdm
import yaml
import pandas as pd 
# load lines
from jiwer import wer

file_topn = "out_hyps_bmsize128.txt"
test_csv = "slurp_csvs/test-type=direct.csv"
table_test = pd.read_csv(test_csv)
real_sentences ={}
for ind, row in table_test.iterrows():
    real_sentences[row["wav"].split("/")[-1]] = row["transcript"]
print(table_test.head)
transcriptions = {}
transcriptions_tokenized = {}
n=5
#Read the topn file :
with open(file_topn, "r") as ft1 : 
    lines = ft1.read().splitlines()
    for line in lines : 
        if line[:2]=="id" : 
            idf = line.split("/")[1]
            transcriptions[idf]=[]
        else : 
            sentence = line.split("/")[1]
            transcriptions[idf].append(sentence)

ref_sentences =[] 
hypothesis = []
for i in real_sentences: 
    ref_sentences.append(real_sentences[i])
    hypothesis.append(transcriptions[i])

def clean(text):
    return ''.join([c for c in text.lower() if c.isalpha() or c == ' ' or c.isdigit()])
references_clean = [clean(r) for r in ref_sentences]
hypothesis_clean = [[clean(x) for x in r] for r in hypothesis]
#hypothesis_clean = [x[1:] for x in hypothesis_clean if x[0]==" "]
partial = False
if partial : 
    end = 10
else : 
    end = len(references_clean)
for i in range(len(references_clean[:10])):
    print("new")
    print(references_clean[i])
    print(hypothesis_clean[i])
wers = []
best_wers = []
for i in range(len(references_clean[:end])) : 
    ref_wers = [wer([references_clean[i]], [hypothesis_clean[i][j]]) for j in range(n)]
    #print(ref_wers)
    wers.append(np.min(ref_wers))
    best_wers.append(ref_wers[0])
print(f"oracle wer from {n}-best : {np.mean(wers)}")
print(f"wer from 1-best : {np.mean(best_wers)}")


