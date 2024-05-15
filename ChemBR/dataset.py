import torch
from torch.utils.data import Dataset
import vocab
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)

def mask_tokens(inputs, stoi, rl):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    inputs = [stoi[s] for s in inputs]
    inputs = torch.tensor(inputs)

    labels = inputs.clone()
    tf = torch.full(labels.shape, 0.15)
    # tf[rl-1] = 1.00

    masked_indices = torch.bernoulli(tf).bool()
    labels[~masked_indices] = -1

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = vocab.smi_vocab.index('<Mask>')

    # 不包含<Start>, <Pad>, <Mask>，故取 3
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(3, vocab.smi_vocab_size, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels

def smiles_to_vocab(smiles):
  smiles_vocab_coding = []
  smiles_length = len(smiles)
  i = 0
  while i < smiles_length:
    if smiles[i] == '[':
      if smiles.startswith(vocab.smi_one_bracket, i):
        slice_end_index = smiles.find(']', i) + 1
        smiles_vocab_coding.append(smiles[i:slice_end_index])
        i = slice_end_index
      elif smiles.startswith(vocab.smi_two_bracket, i):
        slice_end_index = smiles.find(']', i) + 1
        smiles_vocab_coding.append(smiles[i:slice_end_index])
        i = slice_end_index
      elif smiles.startswith(vocab.smi_three_bracket, i):
        slice_end_index = smiles.find(']', i) + 1
        smiles_vocab_coding.append(smiles[i:slice_end_index])
        i = slice_end_index
      elif smiles.startswith(vocab.smi_four_bracket, i):
        slice_end_index = smiles.find(']', i) + 1
        smiles_vocab_coding.append(smiles[i:slice_end_index])
        i = slice_end_index
      else:
        smiles_vocab_coding.append('[')
        i += 1
    elif smiles.startswith(vocab.smi_three_char, i):
        smiles_vocab_coding.append(smiles[i:(i + 3)])
        i += 3
    elif smiles.startswith(vocab.smi_two_char, i):
        smiles_vocab_coding.append(smiles[i:(i + 2)])
        i += 2
    elif smiles.startswith(vocab.smi_one_char, i):
        smiles_vocab_coding.append(smiles[i])
        i += 1
    else:
      return None

  return smiles_vocab_coding

class SmileDataset(Dataset):
    def __init__(self, data, ring_left, ring_right, ring_label, ring_dis):
        # data = [smiles_to_vocab(smi) for smi in data]
        # # print(max(len(sublist) for sublist in data) + 1)
        self.stoi = {ch: i for i, ch in enumerate(vocab.smi_vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab.smi_vocab)}
        self.max_len = 120
        self.vocab_size = vocab.smi_vocab_size
        self.data = data
        self.ring_left = ring_left
        self.ring_right = ring_right
        self.ring_label = ring_label
        self.ring_dis = ring_dis
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles, rl, rr, rlb, rdis = self.data[idx], self.ring_left[idx], self.ring_right[idx], self.ring_label[idx], self.ring_dis[idx]

        smiles = smiles_to_vocab(smiles)

        smiles, label = mask_tokens(smiles, self.stoi, rl)

        start_value = torch.tensor([vocab.smi_vocab.index('<Start>')])
        pad_value = torch.tensor([vocab.smi_vocab.index('<Pad>')])
        ignore_value = torch.tensor([-1])

        smiles = torch.cat((start_value, smiles), dim=0)
        label = torch.cat((ignore_value, label), dim=0)

        if self.max_len - len(smiles) != 0:
            pad_value_length = torch.cat([pad_value] * (self.max_len - len(smiles)))
            ignore_value_length = torch.cat([ignore_value] * (self.max_len - len(smiles)))

            smiles = torch.cat((smiles, pad_value_length), dim=0)
            label = torch.cat((label, ignore_value_length), dim=0)

        if rr - rl == rdis:
            ring_targets = [-1] * rl + [1] * rdis + [-1] * (self.max_len - rl -rdis)
            ring_targets = torch.tensor(ring_targets, dtype=torch.long)
        else:
            ring_targets = [-1] * rl + [0] * rdis + [-1] * (self.max_len - rl -rdis)
            ring_targets = torch.tensor(ring_targets, dtype=torch.long)

        rl = torch.tensor(rl, dtype=torch.long)
        rr = torch.tensor(rr, dtype=torch.long)
        rlb = torch.tensor(rlb, dtype=torch.long)
        rdis = torch.tensor(rdis, dtype=torch.long)

        return smiles, label, rl, rr, rlb, ring_targets, rdis
