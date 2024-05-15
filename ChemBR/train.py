import pandas as pd
from ring_model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import vocab
import random
import numpy as np
import torch
from rdkit import Chem
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    set_seed(42)

    run_name = 'bsmt_chembl_egfr'
    max_epochs = [2, 5, 8, 10][0]
    batch_size = [128, 256, 512][0]

    n_layer = 6
    n_head = 8
    n_embd = 256
    learning_rate = 6e-4

    data = pd.read_csv('datasets/fine_egfr_ring.csv')
    data.columns = data.columns.str.lower()

    train_data = data[data['split'] == 'train'].reset_index(drop=True)
    valid_data = data[data['split'] == 'valid'].reset_index(drop=True)
    test_data = data[data['split'] == 'test'].reset_index(drop=True)

    smiles = train_data['smiles']
    vsmiles = valid_data['smiles']
    tsmiles = test_data['smiles']

    ring_left = train_data['ring_left']
    vring_left = valid_data['ring_left']
    tring_left = test_data['ring_left']

    ring_right = train_data['ring_right']
    vring_right = valid_data['ring_right']
    tring_right = test_data['ring_right']

    ring_label = train_data['ring_label']
    vring_label = valid_data['ring_label']
    tring_label = test_data['ring_label']

    ring_dis = train_data['ring_dis']
    vring_dis = valid_data['ring_dis']
    tring_dis = test_data['ring_dis']

    train_dataset = SmileDataset(smiles, ring_left, ring_right, ring_label, ring_dis)
    valid_dataset = SmileDataset(vsmiles, vring_left, vring_right, vring_label, vring_dis)
    test_dataset = SmileDataset(tsmiles, tring_left, tring_right, tring_label, tring_dis)


    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, n_layer=n_layer,
                      n_head=n_head, n_embd=n_embd, batch_size=batch_size, train_or_generate='train')

    model = GPT(mconf)


    # model.load_state_dict(torch.load('weights/bsmt_chembl.pt'))
    # model.to('cuda')
    # model.tok_emb.weight.requires_grad = False
    # model.type_emb.weight.requires_grad = False
    # model.pos_emb.requires_grad = False
    # model.distance.fc1.weight.requires_grad = False
    # model.distance.fc1.bias.requires_grad = False
    # model.distance.fc2.weight.requires_grad = False
    # model.distance.fc2.bias.requires_grad = False
    # for layer in model.blocks[:5]:
    #     for param in layer.parameters():
    #         param.requires_grad = False
    # for name, param in model.named_parameters():
    #     print(f'Parameter: {name}, Requires gradient: {param.requires_grad}')


    tconf = TrainerConfig(
        max_epochs=max_epochs, batch_size=batch_size, learning_rate=learning_rate, lr_decay=True,
        warmup_tokens=0, final_tokens=max_epochs*len(train_data)*train_dataset.max_len,
        num_workers=10, ckpt_path=f'weights/{run_name}.pt', block_size=train_dataset.max_len)

    trainer = Trainer(model, train_dataset, valid_dataset, test_dataset,
                      tconf, train_dataset.stoi, train_dataset.itos)

    trainer.train()