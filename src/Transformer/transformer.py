"""
完整的 Transformer 模型 (Encoder-Decoder 架构)
"""

import torch
import torch.nn as nn
from typing import Optional

from encoder import Encoder
from decoder import Decoder
from attention import create_padding_mask, create_causal_mask


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_size: int = 512,
        num_heads: int = 8,
        ff_hidden_size: int = 2048,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            ff_hidden_size=ff_hidden_size,
            num_layers=num_encoder_layers,
            max_len=max_len,
            dropout=dropout,
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            ff_hidden_size=ff_hidden_size,
            num_layers=num_decoder_layers,
            max_len=max_len,
            dropout=dropout,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: [batch, src_seq_len] 源序列
            tgt: [batch, tgt_seq_len] 目标序列
        Returns:
            output: [batch, tgt_seq_len, vocab_size]
        """
        src_mask = create_padding_mask(src, self.pad_id)
        encoder_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, encoder_output, None, src_mask)
        return output

    def generate(
        self, src: torch.Tensor, max_len: int = 50, bos_id: int = 2, eos_id: int = 3
    ) -> torch.Tensor:
        """自回归生成"""
        self.eval()
        batch_size = src.size(0)
        device = src.device

        src_mask = create_padding_mask(src, self.pad_id)
        encoder_output = self.encoder(src, src_mask)

        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            output = self.decoder(generated, encoder_output, None, src_mask)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break

        return generated


if __name__ == "__main__":
    print("=" * 60)
    print("Transformer 测试")
    print("=" * 60)

    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        embed_size=64,
        num_heads=4,
        ff_hidden_size=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    src = torch.randint(1, 1000, (2, 10))
    tgt = torch.randint(1, 1000, (2, 8))

    output = model(src, tgt)
    print(f"输出形状: {output.shape}")

    with torch.no_grad():
        generated = model.generate(src, max_len=15)
    print(f"生成序列: {generated[0].tolist()}")
