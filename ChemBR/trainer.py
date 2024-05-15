import math
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    ckpt_path = None
    num_workers = 0

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, test_dataset, config, stoi, itos):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = 'cpu'
        self.stoi = stoi
        self.itos = itos

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model = self.model.to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)

            data = self.train_dataset if is_train else self.valid_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, rl, rr, rlb, rt, rdis) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                rl = rl.to(self.device)
                rr = rr.to(self.device)
                rlb = rlb.to(self.device)
                rt = rt.to(self.device)
                rdis = rdis.to(self.device)

                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        loss, ring = model(x, y, rl, rr, rlb, rt, rdis)

                        loss = loss.mean()
                        losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.3f}. lr {lr:e}."
                                         f"ring {ring:.3f}. ")
            if is_train:
                return float(np.mean(losses))

            if not is_train:
                valid_loss = float(np.mean(losses))
                logger.info(" valid_loss: %f", valid_loss)
                return valid_loss

        best_loss = float('inf')
        self.tokens = 0

        for epoch in range(config.max_epochs):
            train_loss = run_epoch('train')
            if self.valid_dataset is not None:
                valid_loss = run_epoch('valid')

            good_model = self.valid_dataset is None or valid_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                print('best_loss:', best_loss, '  valid_loss:', valid_loss)
                best_loss = valid_loss
                print(f'Saving at epoch {epoch + 1}')
                self.save_checkpoint()