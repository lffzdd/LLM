import torch
import torch.nn as nn
from torch import Tensor


class SelfMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, head_num: int = 8) -> None:
        super().__init__()
        assert embed_dim == head_dim * head_num

        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.head_num = head_num
        self.scale = head_dim**0.5

        self.Q = nn.Linear(embed_dim, head_dim * head_num)
        self.K = nn.Linear(embed_dim, head_dim * head_num)
        self.V = nn.Linear(embed_dim, head_dim * head_num)

        self.dropout = nn.Dropout(0.1)
        self.output_layer = nn.Linear(head_num * head_dim, embed_dim)

    def forward(self, x: Tensor, mask=None):
        # x:[batch_size,seq_len,embed_dim]
        batch_size, seq_len, _ = x.shape

        # [batch_size,seq_len,embed_dim]->[batch_size,seq_len,head_num,head_dim]
        #                               ->[batch_size,head_num,seq_len,head_dim]
        q = (
            self.Q(x)
            .view(batch_size, seq_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.K(x)
            .view(batch_size, seq_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.V(x)
            .view(batch_size, seq_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )

        # scores:[batch_size,head_num,seq_len,seq_len]
        scores: Tensor = q @ k.transpose(-1, -2) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)

        scores = scores.softmax(-1)
        scores = self.dropout(scores)  # dropout 应用于 attention weights

        # output:[batch_size,head_num,seq_len,head_dim]->[batch_size,seq_len,head_num,head_dim]
        #                                              ->[batch_size,seq_len,embed_dim]
        output = scores @ v
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        output = self.output_layer(output)

        return output

class CrossMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, head_num: int = 8) -> None:
        super().__init__()
        assert embed_dim == head_dim * head_num

        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.head_num = head_num
        self.scale = head_dim**0.5

        self.Q = nn.Linear(embed_dim, head_dim * head_num)
        self.K = nn.Linear(embed_dim, head_dim * head_num)
        self.V = nn.Linear(embed_dim, head_dim * head_num)

        self.dropout = nn.Dropout(0.1)
        self.output_layer = nn.Linear(head_num * head_dim, embed_dim)

    def forward(self, x: Tensor, mask):
        # x:[batch_size,seq_len,embed_dim]
        batch_size, seq_len, _ = x.shape

        # [batch_size,seq_len,embed_dim]->[batch_size,seq_len,head_num,head_dim]
        #                               ->[batch_size,head_num,seq_len,head_dim]
        q = (
            self.Q(x)
            .view(batch_size, seq_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.K(x)
            .view(batch_size, seq_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.V(x)
            .view(batch_size, seq_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )

        # scores:[batch_size,head_num,seq_len,seq_len]
        scores: Tensor = q @ k.transpose(1, 2) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)

        scores = scores.softmax(-1)
        scores = self.dropout(scores)  # dropout 应用于 attention weights

        # output:[batch_size,head_num,seq_len,head_dim]->[batch_size,seq_len,head_num,head_dim]
        #                                              ->[batch_size,seq_len,embed_dim]
        output = scores @ v
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        output = self.output_layer(output)

        return output


def create_scores_mask(seq_len: int, device: str) -> Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.to(device)
