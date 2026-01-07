import torch
import torch.nn as nn
from my_transformer._1_tokenizer import BPETokenizer

from torch.utils.data import DataLoader, Dataset


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
