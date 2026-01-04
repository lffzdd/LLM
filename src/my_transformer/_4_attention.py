import torch
import torch.nn as nn
from torch import Tensor


class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim: int, k_dim: int, max_seq_len: int = 512, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim
        self.k_dim = k_dim
        self.scale = k_dim**0.5

        self.w_q = nn.Linear(embed_dim, k_dim)
        self.w_k = nn.Linear(embed_dim, k_dim)
        self.w_v = nn.Linear(embed_dim, k_dim)

        self.w_o = nn.Linear(k_dim, embed_dim)

        # scores_mask = torch.tril(torch.ones((max_seq_len, max_seq_len)))
        scores_mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len), diagonal=1
        ).bool()

        self.register_buffer("scores_mask", scores_mask)

    def forward(self, x: Tensor):
        batch_size, seq_len, embed_dim = x.shape

        Q = self.w_q(x)
        # 等价于 x @ self.w_q.weight.T + self.w_q.bias , 得到 [batch_size, seq_len, k_dim]
        K: Tensor = self.w_k(x)
        V = self.w_v(x)

        # 计算注意力分数,点积加上除以标准差
        scores = Q @ K.transpose(-2, -1) / self.scale

        # 不需要的分数掩码处理
        scores = scores.masked_fill(self.scores_mask[:seq_len, :seq_len], -1e9)
        scores = scores.softmax(dim=-1)

        output = scores @ V
        output = self.w_o(output)

        return output
