import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, heads, scale):
        super().__init__()
        self.h = heads
        self.scale = scale

    def forward(self, q, k, n):
        sim = torch.einsum("b h n l d, b h m d -> b h n l m", q, k) * self.scale

        # masking, avoiding attention within same gene
        # context_mask = F.pad(context_mask, (1, 0), value = True)
        context_mask = torch.ones(n, n + 1).bool().cuda()
        context_mask.fill_diagonal_(False)

        context_mask = rearrange(context_mask, "n m -> 1 1 n 1 m")
        sim = sim.masked_fill(~context_mask, -torch.finfo(sim.dtype).max)

        # max_logits = torch.max(sim, dim=-1, keepdim=True).values
        # sim = sim - max_logits

        # layer_norm = nn.LayerNorm(sim.size()[-1])
        # sim = layer_norm(sim)
        # attention
        attn = sim.softmax(dim=-1)

        return attn
