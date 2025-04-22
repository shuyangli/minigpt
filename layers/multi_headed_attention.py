import torch

class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadedAttention, self).__init__()

    def forward(self, X):
        return X