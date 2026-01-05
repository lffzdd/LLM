import torch
import torch.nn as nn
from torch import Tensor

from my_transformer._3_positional_encoding import PositionalEncoder
from src.my_transformer._2_embedding import TokenEmbedding
from src.my_transformer._4_attention import MultiHeadSelfAttention, create_scores_mask


class EncoderLayer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        attn_head_num: int,
        attn_dim: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.seq_len = seq_len

        self.attn = MultiHeadSelfAttention(embed_dim, attn_head_num, attn_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor):
        # x:[batch_size,seq_len,embed_dim]

        mask = create_scores_mask(self.seq_len, x.device)
        attn_output = self.attn(x, mask)
        x = self.norm1(x + attn_output)

        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class Encoder(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int, max_seq_len: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoder(max_seq_len, embed_dim)
