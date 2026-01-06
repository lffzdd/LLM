import torch
import torch.nn as nn
from torch import Tensor

from src.my_transformer._3_positional_encoding import PositionalEncoder
from src.my_transformer._2_embedding import TokenEmbedding
from src.my_transformer._4_attention import MultiHeadSelfAttention


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attn_head_num: int,
        attn_dim: int,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.attn = MultiHeadSelfAttention(embed_dim, attn_head_num, attn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None):
        # x:[batch_size,seq_len,embed_dim]

        attn_output = self.attn(x, padding_mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        encoder_layer_num: int,
        attn_head_num: int = 8,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        attn_dim = embed_dim // attn_head_num
        assert embed_dim % attn_head_num == 0, (
            "embed_dim must be divisible by attn_head_num"
        )

        self.embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoder(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(embed_dim, attn_head_num, attn_dim, dropout)
                for _ in range(encoder_layer_num)
            ]
        )

    def forward(self, x: Tensor, padding_mask: Tensor | None = None):
        # x:[batch_size, seq_len] -> [batch_size, seq_len, embed_dim] 这里的seq序列已经被pad填充过了
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, padding_mask)

        return x
