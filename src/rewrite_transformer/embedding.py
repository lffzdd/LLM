import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(vocab_size, embed_dim))

    def forward(self, x: Tensor):
        return self.w[x]

nn.Transformer