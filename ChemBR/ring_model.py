import math
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import vocab
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
########################################################################################################
def get_attn_pad_mask(seq_q):
    batch_size, seq_len = seq_q.size()
    pad_attn_mask = seq_q.data.eq(1).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)

class GPTConfig:
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head

    def forward(self, x, attn_mask):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        att = att.masked_fill(attn_mask, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x, att_mask):
        x = x + self.attn(self.ln1(x), att_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # input embedding stem
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

        self.distance = MLP(2, config.n_embd, config.n_embd)

        self.drop = nn.Dropout(0.1)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_mr = nn.LayerNorm(config.n_embd)
        self.head_mr = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.ln_rb = nn.LayerNorm(config.n_embd*2)
        self.head_rb = nn.Linear(config.n_embd*2, 2, bias=False)

        self.block_size = config.block_size

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias') or ('bias' in pn) or fpn.startswith('my_float_params'):
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        decay.add('pos_emb')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, rl=None, rr=None, rlb=None, rt=None, rdis=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        pos_idx = idx.clone()
        enc_self_attn_mask = get_attn_pad_mask(pos_idx)

        token_embeddings = self.tok_emb(pos_idx)
        position_embeddings = self.pos_emb[:, :t, :]
        type_embeddings = self.type_emb(torch.zeros((b, t), dtype=torch.long, device=idx.device))

        x = self.drop(token_embeddings + position_embeddings + type_embeddings)

        if self.config.train_or_generate == 'train':
            for layer in self.blocks:
                x = layer(x, enc_self_attn_mask)

            mr_logits = self.ln_mr(x)
            mr_logits = self.head_mr(mr_logits)

            mr_loss = None
            if targets is not None:
                mr_loss = F.cross_entropy(mr_logits.reshape(-1, mr_logits.size(-1)), targets.view(-1), ignore_index=-1)

            rl = torch.unsqueeze(rl, dim=1)
            rdis = torch.unsqueeze(rdis, dim=1)

            rl_dis = torch.cat((rl, rdis), dim=1).to(torch.float)

            dis = self.distance(rl_dis)
            dis = dis.reshape(dis.size(0), 1, dis.size(1)).repeat(1, self.config.block_size, 1)
            result = torch.cat((x, dis), dim=2)

            po_logits = self.ln_rb(result)
            po_logits = self.head_rb(po_logits)

            po_loss = F.cross_entropy(po_logits.reshape(-1, po_logits.size(-1)), rt.view(-1), ignore_index=-1)

            loss = mr_loss + po_loss

            ring = 0.

            return loss, ring
        if self.config.train_or_generate == 'generate':
            results = np.full((500, 8), False, dtype=bool)

            for layer in self.blocks:
                x = layer(x, enc_self_attn_mask)

            rl = np.array(rl)
            nonzero_values = rl[rl != 0]
            nonzero_indices = np.argwhere(rl != 0)

            if len(nonzero_values) != 0:
                xs = []
                rls = nonzero_values

                for i in range(len(nonzero_values)):
                    open_smi_i = nonzero_indices[i][0]
                    open_smi = x[open_smi_i, :, :]
                    xs.append(open_smi)

                xs = torch.stack(xs)

                rlss = torch.unsqueeze(torch.tensor(rls), dim=1)
                rdiss = rdis - rlss

                rl_dis = torch.cat((rlss, rdiss), dim=1).to(torch.float).to(idx.device)
                dis = self.distance(rl_dis)
                dis = dis.reshape(dis.size(0), 1, dis.size(1)).repeat(1, t, 1)

                result = torch.cat((xs, dis), dim=2)

                po_logits = self.ln_rb(result)
                po_logits = self.head_rb(po_logits)

                # for i in range(len(nonzero_values)):
                #     rlsi = rls[i]
                #     po_logitsi = po_logits[i, rlsi:, :]
                #
                #     po_logitsi_num = po_logitsi[:, 1] > 5*0.8
                #
                #     true_count = po_logitsi_num.sum().item()
                #
                #     if true_count >= len(po_logitsi_num)*0.8:
                #         nonzero_indices_i = nonzero_indices[i][0]
                #         nonzero_indices_j = nonzero_indices[i][1]
                #         results[nonzero_indices_i][nonzero_indices_j] = True

                # for i in range(len(nonzero_values)):
                #     rlsi = rls[i]
                #     # po_logitsi = po_logits[i, rlsi, :]
                #
                #     po_logitsi = po_logits[i, rdis-1, :]
                #
                #     if po_logitsi[1] > 5*0.8:
                #         nonzero_indices_i = nonzero_indices[i][0]
                #         nonzero_indices_j = nonzero_indices[i][1]
                #         results[nonzero_indices_i][nonzero_indices_j] = True

                for i in range(len(nonzero_values)):
                    rlsi = rls[i]
                    po_logitsi = po_logits[i, rlsi:, :]

                    if torch.mean(po_logitsi[:, 1]) > 0.5:
                        nonzero_indices_i = nonzero_indices[i][0]
                        nonzero_indices_j = nonzero_indices[i][1]
                        results[nonzero_indices_i][nonzero_indices_j] = True

            return results
