"""
位置编码 (Positional Encoding)

Transformer 没有 RNN 那样的顺序结构，无法感知 token 的位置信息。
位置编码为每个位置生成一个固定的向量，加到 token embedding 上。

原论文使用 sin/cos 函数：
- PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中：
- pos: token 在序列中的位置 (0, 1, 2, ...)
- i:   embedding 维度的索引 (0, 1, 2, ..., d_model/2)
- d_model: embedding 维度
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置编码层

    使用 sin/cos 函数生成位置编码，这种编码有以下优点：
    1. 每个位置的编码是唯一的
    2. 不同位置之间的相对距离是可学习的（通过线性变换）
    3. 可以推广到比训练时更长的序列
    """

    def __init__(self, embed_size: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            embed_size: embedding 维度 (d_model)
            max_len: 支持的最大序列长度
            dropout: dropout 概率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵 [max_len, embed_size]
        pe = torch.zeros(max_len, embed_size)

        # 位置索引 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算分母中的指数项
        # div_term = 10000^(2i/d_model) 的倒数
        # 使用 log 和 exp 来避免数值问题
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )

        # 偶数位置用 sin，奇数位置用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 0, 2, 4, ...
        pe[:, 1::2] = torch.cos(position * div_term)  # 1, 3, 5, ...

        # 增加 batch 维度 [1, max_len, embed_size]
        pe = pe.unsqueeze(0)

        # 注册为 buffer，不参与梯度更新，但会被保存
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        给输入加上位置编码

        Args:
            x: [batch_size, seq_len, embed_size]
        Returns:
            x + positional_encoding: [batch_size, seq_len, embed_size]
        """
        seq_len = x.size(1)
        # 取出对应长度的位置编码，加到输入上
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码

    与固定的 sin/cos 编码不同，这种方式让模型自己学习位置表示。
    GPT、BERT 等模型使用这种方式。

    缺点：无法处理超过 max_len 的序列
    """

    def __init__(self, embed_size: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            embed_size: embedding 维度
            max_len: 支持的最大序列长度
            dropout: dropout 概率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 可学习的位置 embedding
        self.pe = nn.Embedding(max_len, embed_size)

        # 位置索引 [max_len]，注册为 buffer
        self.register_buffer(
            "positions",
            torch.arange(max_len).expand(1, -1),  # [1, max_len]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_size]
        Returns:
            x + positional_embedding: [batch_size, seq_len, embed_size]
        """
        seq_len = x.size(1)
        positions = self.positions[:, :seq_len]  # [1, seq_len]
        x = x + self.pe(positions)  # broadcast to [batch_size, seq_len, embed_size]
        return self.dropout(x)


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("位置编码测试")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    embed_size = 16

    # 创建假的 embedding 输入
    x = torch.randn(batch_size, seq_len, embed_size)
    print(f"输入形状: {x.shape}")

    # 测试 sin/cos 位置编码
    print("\n1. Sin/Cos 位置编码:")
    pe_sincos = PositionalEncoding(embed_size, max_len=100, dropout=0.0)
    output1 = pe_sincos(x)
    print(f"  输出形状: {output1.shape}")
    print(f"  位置编码矩阵形状: {pe_sincos.pe.shape}")

    # 测试可学习位置编码
    print("\n2. 可学习位置编码:")
    pe_learn = LearnablePositionalEncoding(embed_size, max_len=100, dropout=0.0)
    output2 = pe_learn(x)
    print(f"  输出形状: {output2.shape}")
    print(f"  位置 embedding 权重形状: {pe_learn.pe.weight.shape}")

    # 可视化位置编码模式
    print("\n3. 位置编码可视化 (前5个位置，前8个维度):")
    pe_values = pe_sincos.pe[0, :5, :8]
    print(pe_values)

    # 验证不同位置的编码不同
    print("\n4. 验证位置编码唯一性:")
    pos0 = pe_sincos.pe[0, 0, :]
    pos1 = pe_sincos.pe[0, 1, :]
    similarity = torch.cosine_similarity(pos0, pos1, dim=0)
    print(f"  位置 0 和位置 1 的余弦相似度: {similarity:.4f}")
