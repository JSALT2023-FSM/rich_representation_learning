import jiwer 
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
whisper_path = "/gpfsstore/rech/nou/uzn19yk/whisper-small.en/"
processor = WhisperProcessor.from_pretrained(whisper_path)

model = WhisperForConditionalGeneration.from_pretrained(whisper_path).cuda()
model.config.forced_decoder_ids = None
#model = model.to('cuda')
import yaml
import torchaudio
import pandas as pd 
# load lines
audio, wav_path = None, None
sr = 16000
transcriptions = []
references = []
filein = "slurp_csvs/test-type=direct.csv"
table = pd.read_csv(filein)
for ind, row in tqdm(table[:1000].iterrows()):
    path = row["wav"]
    audio, sr = torchaudio.load(path)
    audio = torch.squeeze(audio)
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.cuda()
    references.append(row["transcript"])
    # generate token ids
    predicted_ids = model.generate(input_features, num_beams=5)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    transcriptions.append(transcription[0])
from jiwer import wer
def clean(text):
    return ''.join([c for c in text.lower() if c.isalpha() or c == ' ' or c.isdigit()])
references_clean = [clean(r) for r in references]
hypothesis_clean = [clean(r) for r in transcriptions]
hypothesis_clean = [x[1:] for x in hypothesis_clean if x[0]==" "]
for i in range(len(references)):
    print("new")
    print(references_clean[i])
    print(hypothesis_clean[i])
print('WER:', wer(references_clean, hypothesis_clean))


