"""
注意力机制 (Attention Mechanism)

核心公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

多头注意力 (Multi-Head Attention):
- 将 Q, K, V 分成多个头
- 每个头独立计算注意力
- 最后拼接所有头的输出

自注意力 (Self-Attention): Q = K = V = 输入
交叉注意力 (Cross-Attention): Q 来自解码器，K, V 来自编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    为什么要缩放？
    - 当 d_k 很大时，点积的值会很大
    - 大的值经过 softmax 后梯度会很小
    - 除以 sqrt(d_k) 可以缓解这个问题
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, heads, seq_len, d_k]
            key:   [batch, heads, seq_len, d_k]
            value: [batch, heads, seq_len, d_v]
            mask:  [batch, 1, 1, seq_len] 或 [batch, 1, seq_len, seq_len]
                   值为 True 的位置会被屏蔽
        Returns:
            output: [batch, heads, seq_len, d_v]
            attention_weights: [batch, heads, seq_len, seq_len]
        """
        d_k = query.size(-1)

        # Step 1: 计算注意力分数 QK^T / sqrt(d_k)
        # [batch, heads, seq_len, d_k] @ [batch, heads, d_k, seq_len]
        # -> [batch, heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Step 2: 应用 mask（如果有）
        if mask is not None:
            # 把 mask 为 True 的位置设为很小的负数，softmax 后趋近于 0
            scores = scores.masked_fill(mask, float("-inf"))

        # Step 3: Softmax 归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Step 4: 加权求和 V
        # [batch, heads, seq_len, seq_len] @ [batch, heads, seq_len, d_v]
        # -> [batch, heads, seq_len, d_v]
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
    where head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)

    多头的好处：
    1. 可以关注不同位置的不同表示子空间
    2. 增加模型的表达能力
    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            embed_size: 输入/输出维度 (d_model)
            num_heads: 注意力头数
            dropout: dropout 概率
        """
        super().__init__()

        assert embed_size % num_heads == 0, (
            f"embed_size ({embed_size}) 必须能被 num_heads ({num_heads}) 整除"
        )

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads  # 每个头的维度 d_k = d_v

        # 线性变换层
        self.W_Q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_K = nn.Linear(embed_size, embed_size, bias=False)
        self.W_V = nn.Linear(embed_size, embed_size, bias=False)
        self.W_O = nn.Linear(embed_size, embed_size, bias=False)

        # 注意力计算
        self.attention = ScaledDotProductAttention(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, seq_len, embed_size]
            key:   [batch, seq_len, embed_size]
            value: [batch, seq_len, embed_size]
            mask:  可选的 attention mask
        Returns:
            output: [batch, seq_len, embed_size]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)

        # Step 1: 线性变换
        Q = self.W_Q(query)  # [batch, seq_len, embed_size]
        K = self.W_K(key)
        V = self.W_V(value)

        # Step 2: 分成多个头
        # [batch, seq_len, embed_size] -> [batch, seq_len, num_heads, head_dim]
        # -> [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: 计算注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Step 4: 合并多个头
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        # -> [batch, seq_len, embed_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embed_size)

        # Step 5: 最后的线性变换
        output = self.W_O(attn_output)

        return output, attn_weights


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    创建因果掩码 (Causal Mask)，用于解码器的自注意力

    防止位置 i 看到位置 i+1, i+2, ... 的信息

    返回的 mask 中，True 表示需要屏蔽的位置

    例如 seq_len=4:
    [[False, True,  True,  True ],
     [False, False, True,  True ],
     [False, False, False, True ],
     [False, False, False, False]]
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )
    return mask


def create_padding_mask(seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """
    创建填充掩码 (Padding Mask)

    屏蔽 PAD token，不让模型关注填充位置

    Args:
        seq: [batch, seq_len] token ids
        pad_id: PAD token 的 id
    Returns:
        mask: [batch, 1, 1, seq_len]
              True 表示需要屏蔽的位置 (PAD)
    """
    # [batch, seq_len] -> [batch, 1, 1, seq_len]
    mask = (seq == pad_id).unsqueeze(1).unsqueeze(2)
    return mask


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("注意力机制测试")
    print("=" * 60)

    batch_size = 2
    seq_len = 4
    embed_size = 16
    num_heads = 4

    # 创建输入
    x = torch.randn(batch_size, seq_len, embed_size)
    print(f"输入形状: {x.shape}")

    # 测试多头自注意力
    print("\n1. 多头自注意力:")
    mha = MultiHeadAttention(embed_size, num_heads, dropout=0.0)
    output, attn_weights = mha(x, x, x)
    print(f"  输出形状: {output.shape}")
    print(f"  注意力权重形状: {attn_weights.shape}")

    # 测试因果掩码
    print("\n2. 因果掩码:")
    causal_mask = create_causal_mask(seq_len)
    print(f"  掩码形状: {causal_mask.shape}")
    print(f"  掩码内容:\n{causal_mask.int()}")

    # 带掩码的注意力
    output_masked, attn_weights_masked = mha(x, x, x, mask=causal_mask)
    print(f"\n  带掩码的输出形状: {output_masked.shape}")
    print(f"  注意力权重 (第一个样本，第一个头):\n{attn_weights_masked[0, 0]}")

    # 测试填充掩码
    print("\n3. 填充掩码:")
    tokens = torch.tensor(
        [
            [1, 2, 3, 0],  # 最后一个是 PAD
            [1, 2, 0, 0],
        ]
    )  # 最后两个是 PAD
    pad_mask = create_padding_mask(tokens, pad_id=0)
    print(f"  Token 序列:\n{tokens}")
    print(f"  填充掩码:\n{pad_mask.squeeze()}")
