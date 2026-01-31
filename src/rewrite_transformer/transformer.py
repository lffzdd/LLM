import torch
import torch.nn as nn
from torch import Tensor

from rewrite_transformer.attention import create_causal_mask, create_padding_mask
from rewrite_transformer.decoder import Decoder
from rewrite_transformer.encoder import Encoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim,
        max_seq_len,
        head_dim,
        head_num,
        encoder_layer_num: int = 6,
        decoder_layer_num: int = 6,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.encoder = Encoder(
            src_vocab_size,
            embed_dim,
            max_seq_len,
            head_dim,
            head_num,
            encoder_layer_num,
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            embed_dim,
            max_seq_len,
            head_dim,
            head_num,
            decoder_layer_num,
        )

        self.output_layer = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src_seq: Tensor, tgt_seq: Tensor):
        src_padding_mask = create_padding_mask(src_seq, self.pad_id)

        tgt_seq_len = tgt_seq.shape[1]
        tgt_casual_mask = create_causal_mask(tgt_seq_len, tgt_seq.device)
        tgt_padding_mask = create_padding_mask(tgt_seq, self.pad_id)

        encoder_output = self.encoder(src_seq, src_padding_mask)

        tgt_output = self.decoder(
            tgt_seq, encoder_output, tgt_padding_mask, tgt_casual_mask, src_padding_mask
        )

        output = self.output_layer(tgt_output)

        return output

    def generate(self, src_seq: Tensor, max_len: int, start_id: int, end_id: int):
        src_padding_mask = create_padding_mask(src_seq, self.pad_id)
        encoder_output = self.encoder(src_seq, src_padding_mask)

        batch_size = src_seq.shape[0]
        generated_seq = torch.full(
            (batch_size, 1), start_id, dtype=torch.long, device=src_seq.device
        )

        for _ in range(max_len):
            tgt_seq_len = generated_seq.shape[1]
            tgt_casual_mask = create_causal_mask(tgt_seq_len, generated_seq.device)
            tgt_padding_mask = create_padding_mask(generated_seq, self.pad_id)

            tgt_output = self.decoder(
                generated_seq,
                encoder_output,
                tgt_padding_mask,
                tgt_casual_mask,
                src_padding_mask,
            )

            output = self.output_layer(tgt_output)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

            generated_seq = torch.cat([generated_seq, next_token], dim=1)

            if (next_token == end_id).all():
                break

        return generated_seq
