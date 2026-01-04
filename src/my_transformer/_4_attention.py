import torch
import torch.nn as nn
from torch import Tensor


class SelfAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.w_q = torch.zeros()
