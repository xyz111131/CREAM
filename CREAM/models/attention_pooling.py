import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        # This is the attention scoring function (can be more complex if needed)
        self.attention_score = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (B, L, C)
        attn_weights = self.attention_score(x)         # (B, L, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (B, L, 1), softmax over L

        # Weighted sum of features across L
        pooled = torch.sum(attn_weights * x, dim=1)    # (B, C)
        return pooled

