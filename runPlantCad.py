import sys
import warnings
import os
import gpn.model
import gpn.pipelines
import torch
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from mamba_ssm import Mamba
from Bio import SeqIO, BiopythonDeprecationWarning

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

warnings.simplefilter('ignore', BiopythonDeprecationWarning)

save_to_file = False
filepath = ''
unknown_args = len(sys.argv) - 2

def is_fasta(filepath):
    try:
        with open(filepath, 'r') as file:
            fasta = SeqIO.read(file, 'fasta')
            
            return any(fasta)
        
    except FileNotFoundError:
        print('File ' + filepath + ' does not exist.\n')
        sys.exit(2)
    except ValueError:
        print('File ' + filepath + ' is in the wrong format.\n')
        sys.exit(3)
        
if '-h' in sys.argv or '--help' in sys.argv:
    with open('README', 'r') as file:
        helptext = file.read()
    print(helptext)
    sys.exit(0)
    
if '-f' in sys.argv or '--file' in sys.argv:
    unknown_args -= 1
    save_to_file = True
    
if unknown_args != 0:
    with open('README', 'r') as file:
        helptext = file.read()
    print(helptext)
    sys.exit(1)
    
if is_fasta(sys.argv[-1]):
    filepath = str(sys.argv[-1])

# Test core dependencies
device = 'cuda:0'

# Test PlantCAD model loading
tokenizer = AutoTokenizer.from_pretrained('kuleshov-group/PlantCaduceus_l32')
model = AutoModelForMaskedLM.from_pretrained('kuleshov-group/PlantCaduceus_l32', trust_remote_code=True)
model.to(device)

gpn_pipeline = pipeline("gpn", model=model, tokenizer=tokenizer, trust_remote_code=True, device='cuda:0')

# Example plant DNA sequence (512bp max)
sequence = str(SeqIO.read(filepath, "fasta").seq)
device = 'cuda:0'
# Get embeddings
encoding = tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )

input_ids = encoding["input_ids"].to(device)
with torch.inference_mode():
    outputs = model(input_ids=input_ids, output_hidden_states=True)

embeddings = outputs.hidden_states[-1]
print(f"Embedding shape: {embeddings.shape}")  # [batch_size, seq_len, embedding_dim]

df = gpn_pipeline(sequence, batch_size=8)[0]
print(df)

if (save_to_file):
    os.makedirs("results", exist_ok=True)
    filename = filepath.split('/')[-1].split('.')[0] + "_model_probabilities.csv"
    df.to_csv("results/"+filename, index=False)