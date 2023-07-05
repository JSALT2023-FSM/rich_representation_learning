import jiwer 
from tqdm import tqdm
import yaml
import pandas as pd 
# load lines
from jiwer import wer

file_top1 = "../results/SLURP/baselines/asr_pretraining_relaunch/1994/test-type=direct.csv"
test_csv = "slurp_csvs/test-type=direct.csv"
table_test = pd.read_csv(test_csv)
real_sentences ={}
for ind, row in table_test.iterrows():
    real_sentences[row["wav"].split("/")[-1]] = row["transcript"]
print(table_test.head)
transcriptions = {}
transcriptions_tokenized = {}
table_test = pd.read_csv(file_top1)
print(table_test.head)
for ind, row in table_test.iterrows():
    transcriptions[row["wav"].split("/")[-1]] = row["asr_transcripts"]

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


