import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.vocab_table = nn.Parameter(torch.randn((vocab_size * embed_dim)))

    def forward(self, x):
        return self.vocab_table[x]
