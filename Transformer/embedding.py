import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        # 手写实现词嵌入层
        self.w=nn.Parameter(torch.randn(vocab_size, embed_size))
        
    def forward(self, x):
        return self.w[x]