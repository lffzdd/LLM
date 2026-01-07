import torch
import torch.nn as nn
from torch import Tensor

from my_transformer._2_embedding import TokenEmbedding
from my_transformer._3_positional_encoding import PositionalEncoder
from my_transformer._4_attention import MultiHeadCrossAttention, MultiHeadSelfAttention


class DecoderLayer(nn.Module):
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

        self.mask_attn = MultiHeadSelfAttention(
            embed_dim, attn_head_num, attn_dim, dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn = MultiHeadCrossAttention(
            embed_dim, attn_head_num, attn_dim, dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,  # Decoder 的输入
        enc_output: Tensor,  # ← Encoder 的输出
        tgt_scores_mask: Tensor | None = None,  # 因果掩码（Self-Attention 用）
        tgt_padding_mask: Tensor | None = None,  # Padding 掩码
        src_padding_mask: Tensor | None = None,  # Padding 掩码
    ):
        """
        x:[batch_size, seq_len, embed_dim]
        enc_output:[batch_size, seq_len, embed_dim]
        """

        attn_output = self.mask_attn(
            x, self._combine_masks(tgt_scores_mask, tgt_padding_mask)
        )
        x = self.norm1(x + self.dropout(attn_output))

        cross_output = self.cross_attn(x, enc_output, src_padding_mask)
        x = self.norm2(x + self.dropout(cross_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

    def _combine_masks(self, scores_mask: Tensor, padding_mask: Tensor):
        """
        scores_mask:[batch_size, seq_len, seq_len]
        padding_mask:[batch_size, seq_len]
        """
        if scores_mask is None and padding_mask is None:
            return None
        if scores_mask is None:
            return padding_mask
        if padding_mask is None:
            return scores_mask

        return scores_mask | padding_mask


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        decoder_layer_num: int,
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

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(embed_dim, attn_head_num, attn_dim, dropout)
                for _ in range(decoder_layer_num)
            ]
        )

    def forward(
        self,
        x: Tensor,
        enc_output: Tensor,
        tgt_scores_mask: Tensor | None = None,
        tgt_padding_mask: Tensor | None = None,
        src_padding_mask: Tensor | None = None,
    ):
        """
        x:[batch_size, seq_len]
        enc_output:[batch_size, seq_len, embed_dim]
        """
        # x:[batch_size, seq_len] -> [batch_size, seq_len, embed_dim] 这里的seq序列已经被pad填充过了
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(
                x, enc_output, tgt_scores_mask, tgt_padding_mask, src_padding_mask
            )

        return x
