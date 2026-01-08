import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from my_transformer._1_tokenizer import BPETokenizer

from torch.utils.data import DataLoader, Dataset

from my_transformer._7_transformer import Transformer


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_texts: list[str],
        tgt_texts: list[str],
        src_tokenizer: BPETokenizer,
        tgt_tokenizer: BPETokenizer,
        max_len: int = 128,
    ) -> None:
        super().__init__()

        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_ids = self.src_tokenizer.encode(self.src_texts[idx])[: self.max_len]
        tgt_ids = self.tgt_tokenizer.encode(self.tgt_texts[idx])[: self.max_len]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch: list[tuple[Tensor, Tensor]], pad_id: int = 0):
    src_batch, tgt_batch = zip(*batch)

    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(s) for s in tgt_batch)

    src_padded = torch.full((len(batch), src_max_len), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max_len), pad_id, dtype=torch.long)

    for i, (src, tgt) in enumerate(batch):
        src_padded[i, : len(src)] = src
        tgt_padded[i, : len(tgt)] = tgt

    return src_padded, tgt_padded


class Trainer:
    def __init__(
        self,
        model: Transformer,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self,dataloader:DataLoader):
        self.model.train()
        total_loss=0

        for src,tgt in dataloader:
            src=src.to(self.device)
            tgt=tgt.to(self.device)
            
            #目标输入和标签
            tgt_input=tgt[:, :-1]
            tgt_label=tgt[:, 1:]
            
            self.optimizer.zero_grad()
            output=self.model(src,tgt_input) # [batch_size, tgt_seq_len, vocab_size]
            
            # output:[batch_size, seq_len, vocab_size] → [batch_size × seq_len, vocab_size], reshape中的-1表示自动计算
            # 
            loss=self.criterion(output.reshape(-1,output.size(-1)),tgt_label.reshape(-1))
            
            loss.backward()
            self.optimizer.step()
            total_loss+=loss.item()
            
        return total_loss/len(dataloader)
                    
