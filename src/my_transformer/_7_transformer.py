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
        Args:
            src: [batch, src_seq_len] 源序列
            tgt: [batch, tgt_seq_len] 目标序列
        Returns:
            output: [batch, tgt_seq_len, vocab_size]
        """
        src_padding_mask = create_padding_mask(src_seq, self.pad_id)
        enc_output = self.encoder(src_seq, src_padding_mask)

        tgt_scores_mask = create_scores_mask(tgt_seq.shape[1], tgt_seq.device)
        tgt_padding_mask = create_padding_mask(tgt_seq, self.pad_id)

        output = self.decoder(
            tgt_seq, enc_output, tgt_scores_mask, tgt_padding_mask, src_padding_mask
        )

        output = self.output_layer(output)

        return output

    def generate(self, src_seq: Tensor, bos_id: int, eos_id: int, max_len: int):
        """自回归生成"""
        self.eval()  # 关闭 dropout

        # src_seq:[batch_size, seq_len]
        batch_size = src_seq.shape[0]
        device = src_seq.device

        src_padding_mask = create_padding_mask(src_seq, self.pad_id)
        enc_output = self.encoder(src_seq, src_padding_mask)

        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

        with torch.no_grad():  # 推理时不需要计算梯度
            for _ in range(max_len - 1):
                tgt_seq_len = generated.shape[1]
                tgt_scores_mask = create_scores_mask(tgt_seq_len, device)

                # [batch_size, tgt_seq_len, vocab_size]
                next_token_prob: Tensor = self.output_layer(
                    self.decoder(
                        generated, enc_output, tgt_scores_mask, None, src_padding_mask
                    )
                )

                # [batch_size, vocab_size] -> [batch_size, 1]
                next_token = next_token_prob[:, -1, :].argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

                if (next_token == eos_id).all():
                    break

        return generated
