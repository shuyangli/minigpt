import torch
from .multi_headed_attention import MultiHeadedAttention
from .feed_forward import GPT2FeedForwardNetwork

class GPT2TransformerDecoder(torch.nn.Module):
    def __init__(self, model_dim: int, num_heads: int, feedforward_dim: int):
        super(GPT2TransformerDecoder, self).__init__()

        self.attn = MultiHeadedAttention(embed_dim=model_dim, num_heads=num_heads)
        self.ln_1 = torch.nn.Linear(in_features=model_dim, out_features=model_dim)
        self.mlp = GPT2FeedForwardNetwork(model_dim=model_dim, feedforward_dim=feedforward_dim)
        self.ln_2 = torch.nn.Linear(in_features=model_dim, out_features=model_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input @ c_attn

        # query, key, value = torch.split(input, split_size_or_sections, dim=1)

        tensor = self.attn(query, key, value)
        tensor = self.ln_1(tensor)
        tensor = self.mlp(tensor)
        tensor = self.ln_2(tensor)
        return tensor