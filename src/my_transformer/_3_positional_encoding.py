import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoder(nn.Module):
    """
    位置编码层

    将位置信息注入到词嵌入中，让模型能够感知 token 的位置。

    公式：
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    def __init__(self, max_seq_len: int, embed_dim: int) -> None:
        """
        Args:
            max_seq_len: 最大序列长度（预计算这么长）
            embed_dim: 嵌入维度
        """
        super().__init__()

        # 预计算位置编码（只计算一次）
        pos_encoding: Tensor = self._create_pos_encoding(max_seq_len, embed_dim)

        # register_buffer: 保存为模型的一部分，但不是参数（不需要梯度）
        # 好处：会随模型一起保存/加载，会自动移动到正确的 device
        self.register_buffer("pos_encoding", pos_encoding)

    def _create_pos_encoding(self, max_seq_len: int, embed_dim: int) -> Tensor:
        """创建位置编码矩阵"""
        pos_vec = torch.zeros(max_seq_len, embed_dim)

        for pos in range(max_seq_len):
            for i in range(embed_dim):
                if i % 2 == 0:
                    pos_vec[pos, i] = math.sin(pos / (10000 ** (i / embed_dim)))
                else:
                    pos_vec[pos, i] = math.cos(pos / (10000 ** ((i - 1) / embed_dim)))

        """这是向量化计算方式
        pos=torch.arange(max_seq_len,device=pos_vec.device).unsqueeze(1)
        div_term=10000 ** (torch.arange(0,embed_dim,2,device=pos_vec.device)/embed_dim)

        pos_vec[:,0::2]=torch.sin(pos/div_term)
        pos_vec[:,1::2]=torch.cos(pos/div_term)
        """

        return pos_vec

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 输入嵌入，形状 [batch_size, seq_len, embed_dim]
        Returns:
            加上位置编码后的嵌入，形状不变
        """
        seq_len = x.shape[1]

        # 直接查表，不需要计算！
        return x + self.pos_encoding[:seq_len, :]
