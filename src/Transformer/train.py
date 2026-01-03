"""
Transformer 训练脚本

包含：
- 数据准备
- 训练循环
- 损失计算
- 模型保存
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
import time

from transformer import Transformer
from tokenizer import BPETokenizer


class TranslationDataset(Dataset):
    """机器翻译数据集"""

    def __init__(
        self,
        src_texts: List[str],
        tgt_texts: List[str],
        src_tokenizer: BPETokenizer,
        tgt_tokenizer: BPETokenizer,
        max_len: int = 128,
    ):
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


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0):
    """对 batch 进行 padding"""
    src_batch, tgt_batch = zip(*batch)

    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(t) for t in tgt_batch)

    src_padded = torch.full((len(batch), src_max_len), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max_len), pad_id, dtype=torch.long)

    for i, (src, tgt) in enumerate(batch):
        src_padded[i, : len(src)] = src
        tgt_padded[i, : len(tgt)] = tgt

    return src_padded, tgt_padded


class Trainer:
    """训练器"""

    def __init__(
        self,
        model: Transformer,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        for src, tgt in dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # 目标的输入和标签
            tgt_input = tgt[:, :-1]  # 去掉最后一个 token
            tgt_label = tgt[:, 1:]  # 去掉第一个 token (BOS)

            self.optimizer.zero_grad()
            output = self.model(src, tgt_input)

            # 计算损失
            loss = self.criterion(
                output.reshape(-1, output.size(-1)), tgt_label.reshape(-1)
            )

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)


if __name__ == "__main__":
    print("=" * 60)
    print("Transformer 训练示例")
    print("=" * 60)

    # 示例数据
    src_texts = ["hello world", "how are you", "good morning"]
    tgt_texts = ["你好世界", "你好吗", "早上好"]

    # 创建 tokenizer 并训练
    src_tokenizer = BPETokenizer()
    tgt_tokenizer = BPETokenizer()
    src_tokenizer.train(src_texts, target_vocab_size=100, verbose=False)
    tgt_tokenizer.train(tgt_texts, target_vocab_size=100, verbose=False)

    print(f"源词表大小: {src_tokenizer.vocab_size}")
    print(f"目标词表大小: {tgt_tokenizer.vocab_size}")

    # 创建模型
    model = Transformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        embed_size=64,
        num_heads=4,
        ff_hidden_size=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("\n训练脚本准备完成！")
