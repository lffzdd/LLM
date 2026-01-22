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
        """
        Args:
            x: [batch_size,seq_len,embed_dim]
            mask: [batch_size,seq_len,seq_len],
                若是encoder，只需要填充掩码，
                若是decoder，因果掩码和填充掩码都需要，此时输入的mask应为因果掩码和填充掩码的按位或
        """
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

        # 若为组合掩码：mask[batch_size,1,seq_len,seq_len]
        # 若为填充掩码：mask[batch_size,1,  1,    seq_len]
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

    def forward(
        self,
        decoder_input: Tensor,
        encoder_output: Tensor,
        padding_mask: Tensor | None = None,
    ):
        """
        交叉注意力处于Decoder中，不需要因果掩码，只需要填充掩码
        """
        # x:[batch_size,seq_len,embed_dim]
        batch_size, tgt_seq_len, _ = decoder_input.shape
        _, src_seq_len, _ = encoder_output.shape

        # [batch_size,seq_len,embed_dim]->[batch_size,seq_len,head_num,head_dim]
        #                               ->[batch_size,head_num,seq_len,head_dim]
        q = (
            self.Q(decoder_input)
            .view(batch_size, tgt_seq_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.K(encoder_output)
            .view(batch_size, src_seq_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.V(encoder_output)
            .view(batch_size, src_seq_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )

        # scores:[batch_size,head_num,tgt_seq_len,src_seq_len]
        scores: Tensor = q @ k.transpose(-1, -2) / self.scale

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, -1e4)

        scores = scores.softmax(-1)
        scores = self.dropout(scores)  # dropout 应用于 attention weights

        # output:[batch_size,head_num,tgt_seq_len,src_seq_len]->[batch_size,tgt_seq_len,head_num,src_seq_len]
        #                                              ->[batch_size,tgt_seq_len,embed_dim]
        output = scores @ v
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, tgt_seq_len, self.embed_dim)
        )
        output = self.output_layer(output)

        return output


def create_causal_mask(seq_len: int, device: str| None = None) -> Tensor:
    """创建因果掩码（Causal Mask），用于 Decoder 自注意力
        防止位置 i 看到位置 i+1, i+2, ... 的信息

        返回的 mask 中，True 表示需要屏蔽的位置
        例如 seq_len=4:
        >>> [[False, True,  True,  True ],
        >>>  [False, False, True,  True ],
        >>>  [False, False, False, True ],
        >>>  [False, False, False, False]]

    Args:
        seq_len: 序列长度
        device: 设备

    Returns:
        掩码 [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    if device:
        mask = mask.to(device)
    return mask


def create_padding_mask(seq: Tensor, pad_id):
    """
    seq:         [batch, seq_len]           → [2, 8]
    mask:        [batch, 1, 1, seq_len]     → [2, 1, 1, 8]
    scores:      [batch, head, seq_len, seq_len] → [2, 8, 8, 8]

    广播后:
        mask 的最后一维 [8] 会广播到 scores 的列维度
        mask 的倒数第二维 [1] 会广播到 scores 的所有行

    Args:
        seq: [batch_size,seq_len],这里的seq还没有进入TokenEmbedding
        pad_id: padding id

    Returns:
        mask: [batch_size,1,1,seq_len]



    """
    # seq:[batch_size,seq_len]
    mask = (seq == pad_id).unsqueeze(1).unsqueeze(2)
    #                      ↑ 广播到所有 head
    #                               ↑ 广播到所有 query 行

    return mask
