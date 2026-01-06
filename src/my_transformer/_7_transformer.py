import torch
import torch.nn as nn

from torch import Tensor

from my_transformer._4_attention import create_padding_mask, create_scores_mask
from my_transformer._5_encoder import Encoder
from my_transformer._6_decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        encoder_layer_num: int = 6,
        decoder_layer_num: int = 6,
        attn_head_num: int = 8,
        dropout: float = 0.1,
        pad_id: int = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.pad_id = pad_id

        self.encoder = Encoder(
            vocab_size,
            embed_dim,
            max_seq_len,
            encoder_layer_num,
            attn_head_num,
            dropout,
        )

        self.decoder = Decoder(
            vocab_size,
            embed_dim,
            max_seq_len,
            decoder_layer_num,
            attn_head_num,
            dropout,
        )

        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, src_seq: Tensor, tgt_seq: Tensor):
        """
        src_seq:[batch_size, seq_len]
        """
        src_padding_mask = create_padding_mask(src_seq, self.pad_id)
        enc_output = self.encoder(src_seq, src_padding_mask)

        tgt_scores_mask = create_scores_mask(tgt_seq.shape[1], src_seq.device)
        tgt_padding_mask = create_padding_mask(tgt_seq, self.pad_id)

        output = self.decoder(
            tgt_seq, enc_output, tgt_scores_mask, tgt_padding_mask, src_padding_mask
        )

        output = self.output_layer(output)

        return output
