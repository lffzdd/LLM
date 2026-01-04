import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.vocab_table = nn.Parameter(torch.randn((vocab_size, embed_dim)))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: token IDs, 形状 [batch_size, seq_len]
        Returns:
            嵌入向量, 形状 [batch_size, seq_len, embed_dim]
        """

        return self.vocab_table[x]
