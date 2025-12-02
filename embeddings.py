import einops
import torch
import math
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from einops.layers.torch import Rearrange


class PatchEmbeddings(nn.Module):
    """
    Module that extracts patches and projects them
    """
    def __init__(self, patch_size2: int, patch_dim: int, emb_dim: int):
        super().__init__()
        self.patchify = Rearrange(
            "b c (h p1) (w p2) -> b (h w) c p1 p2",
            # "b c (h p1) (w p2) -> b (p1 p2) c h w",
            p1=patch_size2, p2=patch_size2)

        self.flatten = nn.Flatten(start_dim=2)
        self.proj = nn.Linear(in_features=patch_dim, out_features=emb_dim)
#        print('patch_dim',patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rearrange into patches
        # print('-------patchemb--------')
        x = self.patchify(x)
        # print(('patchify后',x.shape))
        # Flatten patches into individual vectors
        x = self.flatten(x)
        # print(('flatten', x.shape))
        # Project to higher dim
        x = self.proj(x)
        # print(('proj后', x.shape))
        return x


class CLSToken(nn.Module):
    """
    Prepend cls token to each embedding
    """
    def __init__(self, dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        return x


class PositionalEmbeddings(nn.Module):
    """
    Learned positional embeddings
    """
    def __init__(self, num_pos: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(num_pos, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('-------位置编码--------')
        # print('位置编码', self.pos.shape)
        # print('加上位置编码后', (x + self.pos).shape)
        return x + self.pos
