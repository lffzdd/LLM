import torch
import torch.nn as nn
from torch import Tensor

from rewrite_transformer.attention import SelfMultiHeadAttention
from rewrite_transformer.embedding import TokenEmbedding
from rewrite_transformer.positional_encoding import PositionalEncoder


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, head_dim, head_num: int = 8) -> None:
        super().__init__()
        self.self_attn = SelfMultiHeadAttention(embed_dim, head_dim, head_num)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor, src_padding_mask: Tensor | None = None):
        """Encoder阶段，自注意力不需要因果掩码，只需要针对源序列的填充掩码"""
        attn_output = self.self_attn(x, src_padding_mask)
        # x = self.norm1(x + self.dropout(attn_output))
        # Pre-LN:Transformer 原文把LayerNorm放在了残差连接之后，在这个过程中，梯度会被反复缩放，后来发现把LayerNorm放在残差连接之前效果更好
        x = x + self.norm1(self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = x + self.norm2(self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_dim,
        max_seq_len,
        head_dim,
        head_num: int = 8,
        layer_num: int = 6,
    ) -> None:
        """
        Args:
            src_vocab_size: 源语言词表大小
            embed_dim: 词嵌入维度
            max_seq_len: 最大序列长度,即支持的最长序列长度
            head_dim: 注意力头维度
            head_num: 注意力头数量
            layer_num: encoder_layer层数
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(src_vocab_size, embed_dim)
        self.position_encode = PositionalEncoder(max_seq_len, embed_dim)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(embed_dim, head_dim, head_num) for i in range(layer_num)]
        )

    def forward(self, x: Tensor, src_padding_mask: Tensor | None = None):
        """Encoder阶段，自注意力不需要因果掩码，只需要针对源序列的填充掩码"""
        x = self.token_embedding(x)
        x = x + self.position_encode(x)

        for layer in self.encoder_layers:
            x = layer(x, src_padding_mask)

        return x
