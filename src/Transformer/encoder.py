"""
Transformer 编码器 (Encoder)

编码器层结构：
1. Multi-Head Self-Attention
2. Add & Norm (残差连接 + LayerNorm)
3. Feed-Forward Network
4. Add & Norm

完整编码器 = N 个编码器层堆叠
"""

import torch
import torch.nn as nn
from typing import Optional

from attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    前馈神经网络 (Position-wise Feed-Forward Network)

    FFN(x) = max(0, xW1 + b1)W2 + b2

    结构：Linear -> ReLU -> Dropout -> Linear -> Dropout
    """

    def __init__(self, embed_size: int, ff_hidden_size: int, dropout: float = 0.1):
        """
        Args:
            embed_size: 输入/输出维度
            ff_hidden_size: 中间层维度，通常是 embed_size * 4
            dropout: dropout 概率
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_size]
        Returns:
            output: [batch, seq_len, embed_size]
        """
        x = self.fc1(x)  # [batch, seq_len, ff_hidden_size]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch, seq_len, embed_size]
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    """
    单个编码器层

    结构：
    x -> Self-Attention -> Add & Norm -> FFN -> Add & Norm -> output
         ↑__________________|            ↑_______|
               残差连接                    残差连接
    """

    def __init__(
        self, embed_size: int, num_heads: int, ff_hidden_size: int, dropout: float = 0.1
    ):
        """
        Args:
            embed_size: 模型维度 (d_model)
            num_heads: 注意力头数
            ff_hidden_size: FFN 中间层维度
            dropout: dropout 概率
        """
        super().__init__()

        # 多头自注意力
        self.self_attn = MultiHeadAttention(embed_size, num_heads, dropout)

        # 前馈网络
        self.feed_forward = FeedForward(embed_size, ff_hidden_size, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_size]
            mask: 可选的注意力掩码
        Returns:
            output: [batch, seq_len, embed_size]
        """
        # Self-Attention + 残差 + LayerNorm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-Forward + 残差 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class Encoder(nn.Module):
    """
    完整的 Transformer 编码器

    结构：
    Input Embedding + Positional Encoding -> N x EncoderLayer -> Output
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        num_heads: int,
        ff_hidden_size: int,
        num_layers: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: 词表大小
            embed_size: 模型维度
            num_heads: 注意力头数
            ff_hidden_size: FFN 中间层维度
            num_layers: 编码器层数
            max_len: 最大序列长度
            dropout: dropout 概率
        """
        super().__init__()

        self.embed_size = embed_size

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_size)

        # Positional Encoding (使用 sin/cos)
        from positional_encoding import PositionalEncoding

        self.positional_encoding = PositionalEncoding(embed_size, max_len, dropout)

        # 编码器层堆叠
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embed_size, num_heads, ff_hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )

        # 缩放因子：原论文中 embedding 要乘以 sqrt(d_model)
        self.scale = embed_size**0.5

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len] token ids
            mask: 可选的注意力掩码（如 padding mask）
        Returns:
            output: [batch, seq_len, embed_size]
        """
        # Token Embedding + 缩放 + Positional Encoding
        x = self.token_embedding(x) * self.scale
        x = self.positional_encoding(x)

        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, mask)

        return x


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("编码器测试")
    print("=" * 60)

    # 配置
    vocab_size = 1000
    embed_size = 64
    num_heads = 4
    ff_hidden_size = 256
    num_layers = 2
    batch_size = 2
    seq_len = 10

    # 创建编码器
    encoder = Encoder(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        ff_hidden_size=ff_hidden_size,
        num_layers=num_layers,
        dropout=0.1,
    )

    print(f"编码器配置:")
    print(f"  词表大小: {vocab_size}")
    print(f"  模型维度: {embed_size}")
    print(f"  注意力头数: {num_heads}")
    print(f"  FFN 隐藏层: {ff_hidden_size}")
    print(f"  编码器层数: {num_layers}")

    # 计算参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  总参数量: {total_params:,}")

    # 测试前向传播
    print(f"\n输入:")
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"  Token 形状: {tokens.shape}")

    output = encoder(tokens)
    print(f"\n输出:")
    print(f"  输出形状: {output.shape}")

    # 测试带 mask
    print("\n带 Padding Mask 的测试:")
    from attention import create_padding_mask

    tokens_with_pad = torch.tensor([[1, 2, 3, 4, 0, 0], [1, 2, 3, 0, 0, 0]])
    mask = create_padding_mask(tokens_with_pad, pad_id=0)
    output_masked = encoder(tokens_with_pad, mask)
    print(f"  输入形状: {tokens_with_pad.shape}")
    print(f"  输出形状: {output_masked.shape}")
