import torch
import torch.nn as nn
from torch import Tensor

from rewrite_transformer.attention import (
    SelfMultiHeadAttention,
    CrossMultiHeadAttention,
)
from rewrite_transformer.embedding import TokenEmbedding
from rewrite_transformer.positional_encoding import PositionalEncoder


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, head_dim, head_num: int = 8) -> None:
        super().__init__()
        self.self_attn = SelfMultiHeadAttention(embed_dim, head_dim, head_num)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossMultiHeadAttention(embed_dim, head_dim, head_num)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        tgt_padding_mask: Tensor | None = None,
        tgt_casual_mask: Tensor | None = None,
        src_padding_mask: Tensor | None = None,
    ):
        """
        Decoder的自注意力需要因果掩码和填充掩码
        交叉注意力不需要因果掩码，只需要填充掩码
        """

        self_attn_output = self.self_attn(
            x, self._combine_masks(tgt_padding_mask, tgt_casual_mask)
        )
        x = self.norm1(x + self.dropout(self_attn_output))

        cross_attn_output = self.cross_attn(x, encoder_output, src_padding_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        x = self.norm3(x + self.dropout(self.feed_forward(x)))

        return x

    def _combine_masks(self, tgt_padding_mask, tgt_casual_mask):
        """
        tgt_padding_mask:[batch_size,1,1,seq_len]
        tgt_casual_mask:[seq_len,seq_len]

        会广播成 [batch_size,1 ,seq_len,seq_len]
        """
        if tgt_padding_mask is None and tgt_casual_mask is None:
            return None
        if tgt_padding_mask is None:
            return tgt_casual_mask
        if tgt_casual_mask is None:
            return tgt_padding_mask
        return tgt_padding_mask | tgt_casual_mask


class Decoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size,
        embed_dim,
        max_seq_len,
        head_dim,
        head_num: int = 8,
        layer_num: int = 6,
    ) -> None:
        """
        Args:
            tgt_vocab_size: 目标语言词表大小
            embed_dim: 词嵌入维度
            max_seq_len: 最大序列长度,即支持的最长序列长度
            head_dim: 注意力头维度
            head_num: 注意力头数量
            layer_num: encoder_layer层数
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(tgt_vocab_size, embed_dim)
        self.position_encode = PositionalEncoder(max_seq_len, embed_dim)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(embed_dim, head_dim, head_num) for i in range(layer_num)]
        )

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        tgt_padding_mask: Tensor | None = None,
        tgt_casual_mask: Tensor | None = None,
        src_padding_mask: Tensor | None = None,
    ):
        # x:[batch_size,seq_len]->[batch_size,seq_len,embed_dim]
        x = self.token_embedding(x)
        x = x + self.position_encode(x)

        for layer in self.decoder_layers:
            x = layer(
                x, encoder_output, tgt_padding_mask, tgt_casual_mask, src_padding_mask
            )

        return x
