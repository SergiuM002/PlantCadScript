import sys
import warnings
import os
import math
import gpn.model
import gpn.pipelines
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from mamba_ssm import Mamba
from Bio import SeqIO, BiopythonDeprecationWarning

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
        
def sem(probs, probref):
    _sum = 2
    
    for i in range(4):
        _sum += probs[i] * math.log2(probs[i])
    
    return _sum * (probref - 0.25)

def sb(probs, probref):
    _sum = 0
    
    for i in range(4):
        _sum += probs[i] * math.log2(probs[i]) 
        
    return -_sum + math.log2(probref)
    

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', BiopythonDeprecationWarning)

save_to_file = False
filepath = ''
unknown_args = len(sys.argv) - 2

helptext = ''

with open('README', 'r') as file:
        helptext = str(file.read()).split('////')[1]
        
if '-h' in sys.argv or '--help' in sys.argv:
    print(helptext)
    sys.exit(0)
    
if '-f' in sys.argv or '--file' in sys.argv:
    unknown_args -= 1
    save_to_file = True
    
if unknown_args != 0:
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

sequence = str(SeqIO.read(filepath, "fasta").seq)

df = gpn_pipeline(sequence, batch_size=8)[0]
    
sem_scores = []
sb_scores = []

for i in range(len(df)):
    row = df.iloc[i].tolist()
    
    sem_scores.append(sem(row[2:6], row[1]))
    sb_scores.append(sb(row[2:6], row[1]))
    
df["SEM"] = sem_scores
df["SB"] = sb_scores

print(df)

if save_to_file:
    os.makedirs("results", exist_ok=True)
    filename = filepath.split('/')[-1].split('.')[0]
    os.makedirs("results/" + filename, exist_ok=True)
    df.to_csv("results/" + filename + "/model_probabilities.csv", index=False)
    
    sem_plot = df.reset_index().plot.scatter(
        x='index',
        y='SEM',
        s=20/(len(df)/1000),
        c='SEM',
        cmap='PiYG',
        norm=mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=1.5),
        colorbar=False
    )
    
    sb_plot = df.reset_index().plot.scatter(
        x='index',
        y='SB',
        s=20/(len(df)/1000),
        colorbar=False
    )
    
    sem_plot.get_figure().savefig("results/" + filename + "/SEM_plot.png", dpi=150)
    sb_plot.get_figure().savefig("results/" + filename + "/SB_plot.png", dpi=150)