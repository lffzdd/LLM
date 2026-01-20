import torch.nn as nn
import torch
from torch import Tensor, embedding


class PositionalEncoder(nn.Module):
    def __init__(self, max_seq_len, embed_dim) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        pos_vec = self._create_pos_encoding()
        self.register_buffer("pos_vec", pos_vec)

    def _create_pos_encoding(self):
        pos_vec = torch.zeros(self.max_seq_len, self.embed_dim)

        # [max_seq_len, 1] - 增加维度以便广播
        pos = torch.arange(self.max_seq_len).unsqueeze(1)

        # [embed_dim/2]
        div_term = 10000 ** (torch.arange(0, self.embed_dim, 2) / self.embed_dim)

        # pos / div_term 广播: [max_seq_len, 1] / [embed_dim/2] -> [max_seq_len, embed_dim/2]
        pos_vec[:, 0::2] = torch.sin(pos / div_term)
        pos_vec[:, 1::2] = torch.cos(pos / div_term)
        return pos_vec

    def forward(self, x: Tensor):
        # x: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.shape

        # 检查 embed_dim 是否匹配
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"输入的 embed_dim ({embed_dim}) 与 PositionalEncoder 的 embed_dim ({self.embed_dim}) 不匹配"
            )

        # 检查 seq_len 是否超过 max_seq_len
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"输入的 seq_len ({seq_len}) 超过了 max_seq_len ({self.max_seq_len})"
            )

        # 只取前 seq_len 行的位置编码
        # pos_vec[:seq_len] 的 shape: [seq_len, embed_dim]
        # 广播加到 x 上: [batch_size, seq_len, embed_dim] + [seq_len, embed_dim]
        return x + self.pos_vec[:seq_len]
