import jiwer 
from tqdm import tqdm
import yaml
import pandas as pd 
# load lines
from jiwer import wer

file_top1 = "../slurp_top1.txt"
test_csv = "slurp_csvs/test-type=direct.csv"
table_test = pd.read_csv(test_csv)
real_sentences ={}
for ind, row in table_test.iterrows():
    real_sentences[row["wav"].split("/")[-1]] = row["transcript"]
print(table_test.head)
transcriptions = {}
transcriptions_tokenized = {}
with open(file_top1, "r") as ft1 : 
    lines = ft1.read().splitlines()
    for line in lines : 
        split, idf, trans = line.split(":")
        idf = idf.split("/")[1][:-1]
        transcriptions[idf] = trans

ref_sentences =[] 
hypothesis = []
for i in real_sentences : 
    ref_sentences.append(real_sentences[i])
    hypothesis.append(transcriptions[i])

def clean(text):
    return ''.join([c for c in text.lower() if c.isalpha() or c == ' ' or c.isdigit()])
references_clean = [clean(r) for r in ref_sentences]
hypothesis_clean = [clean(r) for r in hypothesis]
#hypothesis_clean = [x[1:] for x in hypothesis_clean if x[0]==" "]
for i in range(len(references_clean[0:10])):
    print("new")
    print(references_clean[i])
    print(hypothesis_clean[i])
print('WER:', wer(references_clean, hypothesis_clean))


