import torch
# from multi_headed_attention import MultiHeadedAttention

class GPT2FeedForwardNetwork(torch.nn.Module):
    def __init__(self, model_dim: int, feedforward_dim: int):
        super(GPT2FeedForwardNetwork, self).__init__()

        # Up projection
        self.c_fc = torch.nn.Linear(in_features=model_dim, out_features=feedforward_dim)
        # Gelu
        self.gelu = torch.nn.GELU()
        # Down projection
        self.c_proj = torch.nn.Linear(in_features=feedforward_dim, out_features=model_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.c_fc(input)
        output = self.gelu(output)
        output = self.c_proj(output)
        return output