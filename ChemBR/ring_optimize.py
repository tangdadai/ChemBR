import random
import numpy as np
import torch
from torch.nn import functional as F
from rdkit import Chem
import vocab
import pandas as pd
from dataset import smiles_to_vocab

from ring_model import GPT, GPTConfig
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

@torch.no_grad()
def ring_sample(x, posi, k):
    model_weight = 'bsmt_chembl_egfr.pt'
    vocab_size = 100
    block_size = 120
    n_layer = 6
    n_head = 8
    n_embd = 256

    mconf = GPTConfig(vocab_size, block_size, n_layer=n_layer, n_head=n_head,
                      n_embd=n_embd, train_or_generate='generate')
    model = GPT(mconf)

    model.load_state_dict(torch.load('weights/' + model_weight))
    model.to('cuda')

    position_class = model(x, rl=posi, rdis=k)
    return position_class
