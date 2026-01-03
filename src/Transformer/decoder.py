"""
Transformer 解码器 (Decoder)

解码器层结构（比编码器多一层交叉注意力）：
1. Masked Multi-Head Self-Attention (带因果掩码)
2. Add & Norm
3. Multi-Head Cross-Attention (查询来自解码器，键值来自编码器)
4. Add & Norm
5. Feed-Forward Network
6. Add & Norm

完整解码器 = N 个解码器层堆叠
"""

import torch
import torch.nn as nn
from typing import Optional

from attention import MultiHeadAttention, create_causal_mask
from encoder import FeedForward


class DecoderLayer(nn.Module):
    """
    单个解码器层

    结构：
    x -> Masked Self-Attn -> Add & Norm -> Cross-Attn -> Add & Norm -> FFN -> Add & Norm
         ↑__________________|              ↑___________|              ↑_____|
              残差连接                        残差连接                  残差连接
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

        # Masked Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(embed_size, num_heads, dropout)

        # Cross-Attention (用于关注编码器输出)
        self.cross_attn = MultiHeadAttention(embed_size, num_heads, dropout)

        # 前馈网络
        self.feed_forward = FeedForward(embed_size, ff_hidden_size, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, tgt_seq_len, embed_size] 解码器输入
            encoder_output: [batch, src_seq_len, embed_size] 编码器输出
            self_attn_mask: 自注意力掩码（因果掩码 + padding掩码）
            cross_attn_mask: 交叉注意力掩码（源序列的padding掩码）
        Returns:
            output: [batch, tgt_seq_len, embed_size]
        """
        # 1. Masked Self-Attention + 残差 + LayerNorm
        self_attn_output, _ = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # 2. Cross-Attention + 残差 + LayerNorm
        # Q 来自解码器，K, V 来自编码器
        cross_attn_output, _ = self.cross_attn(
            x, encoder_output, encoder_output, cross_attn_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. Feed-Forward + 残差 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)

        return x


class Decoder(nn.Module):
    """
    完整的 Transformer 解码器

    结构：
    Target Embedding + Positional Encoding -> N x DecoderLayer -> Linear -> Output
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
            num_layers: 解码器层数
            max_len: 最大序列长度
            dropout: dropout 概率
        """
        super().__init__()

        self.embed_size = embed_size

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_size)

        # Positional Encoding
        from positional_encoding import PositionalEncoding

        self.positional_encoding = PositionalEncoding(embed_size, max_len, dropout)

        # 解码器层堆叠
        self.layers = nn.ModuleList(
            [
                DecoderLayer(embed_size, num_heads, ff_hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )

        # 最终的线性层：映射回词表大小
        self.output_projection = nn.Linear(embed_size, vocab_size)

        # 缩放因子
        self.scale = embed_size**0.5

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: [batch, tgt_seq_len] 目标序列 token ids
            encoder_output: [batch, src_seq_len, embed_size] 编码器输出
            self_attn_mask: 自注意力掩码
            cross_attn_mask: 交叉注意力掩码
        Returns:
            output: [batch, tgt_seq_len, vocab_size] 每个位置的词表概率分布
        """
        # Token Embedding + 缩放 + Positional Encoding
        x = self.token_embedding(tgt) * self.scale
        x = self.positional_encoding(x)

        # 如果没有提供自注意力掩码，创建因果掩码
        if self_attn_mask is None:
            tgt_seq_len = tgt.size(1)
            self_attn_mask = create_causal_mask(tgt_seq_len, device=tgt.device)

        # 通过所有解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)

        # 投影到词表空间
        output = self.output_projection(x)

        return output


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("解码器测试")
    print("=" * 60)

    # 配置
    vocab_size = 1000
    embed_size = 64
    num_heads = 4
    ff_hidden_size = 256
    num_layers = 2
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8

    # 创建解码器
    decoder = Decoder(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        ff_hidden_size=ff_hidden_size,
        num_layers=num_layers,
        dropout=0.1,
    )

    print(f"解码器配置:")
    print(f"  词表大小: {vocab_size}")
    print(f"  模型维度: {embed_size}")
    print(f"  注意力头数: {num_heads}")
    print(f"  FFN 隐藏层: {ff_hidden_size}")
    print(f"  解码器层数: {num_layers}")

    # 计算参数量
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"  总参数量: {total_params:,}")

    # 模拟编码器输出
    encoder_output = torch.randn(batch_size, src_seq_len, embed_size)
    print(f"\n编码器输出形状: {encoder_output.shape}")

    # 目标序列
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
    print(f"目标序列形状: {tgt_tokens.shape}")

    # 前向传播
    output = decoder(tgt_tokens, encoder_output)
    print(f"解码器输出形状: {output.shape}")  # [batch, tgt_seq_len, vocab_size]

    # 获取预测的 token
    predicted_tokens = output.argmax(dim=-1)
    print(f"预测 Token 形状: {predicted_tokens.shape}")
    print(f"预测 Token 示例: {predicted_tokens[0]}")
