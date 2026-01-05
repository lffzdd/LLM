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


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_num: int,
        k_dim: int,
        max_seq_len: int = 512,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim
        self.head_num = head_num
        self.k_dim = k_dim
        self.scale = k_dim**0.5

        self.w_q = nn.Linear(embed_dim, head_num * k_dim)
        self.w_k = nn.Linear(embed_dim, head_num * k_dim)
        self.w_v = nn.Linear(embed_dim, head_num * k_dim)

        self.w_o = nn.Linear(head_num * k_dim, embed_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None):
        batch_size, seq_len, _ = x.shape

        Q: Tensor = self.w_q(x)  # [batch_size, seq_len, k_dim * head_num]
        K: Tensor = self.w_k(x)
        V: Tensor = self.w_v(x)

        # ========== 分头 ==========
        # 步骤 1: view 拆分最后一维
        # [batch, seq_len, head_num * k_dim] → [batch, seq_len, head_num, k_dim]
        Q = Q.view(batch_size, seq_len, self.head_num, self.k_dim)
        K = K.view(batch_size, seq_len, self.head_num, self.k_dim)
        V = V.view(batch_size, seq_len, self.head_num, self.k_dim)

        # 步骤 2: transpose 把 head_num 移到前面
        # [batch, seq_len, head_num, k_dim] → [batch, head_num, seq_len, k_dim]
        Q = Q.transpose(1, 2)  # 交换维度 1 和 2
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ========== 计算注意力（每个头独立计算）==========
        # [batch, head_num, seq_len, k_dim] @ [batch, head_num, k_dim, seq_len]
        # → [batch, head_num, seq_len, seq_len]
        scores = Q @ K.transpose(-2, -1) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        # [batch, head_num, seq_len, seq_len] @ [batch, head_num, seq_len, k_dim]
        # → [batch, head_num, seq_len, k_dim]
        output = scores @ V

        # [batch, head_num, seq_len, k_dim] → [batch, seq_len, head_num * k_dim]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.head_num * self.k_dim)

        # ========== 线性变换 ==========
        output = self.w_o(output)

        return output


def create_scores_mask(seq_len: int, device: torch.device = None) -> Tensor:
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
