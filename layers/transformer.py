import torch
from .multi_headed_attention import MultiHeadedAttention
from .feed_forward import GPT2FeedForwardNetwork
from params.model_params import TransformerBlockParams

class GPT2TransformerDecoder(torch.nn.Module):
    def __init__(self, model_dim: int, num_heads: int, feedforward_dim: int):
        super(GPT2TransformerDecoder, self).__init__()

        self.attn = MultiHeadedAttention(embed_dim=model_dim, num_heads=num_heads)
        self.ln_1 = torch.nn.Linear(in_features=model_dim, out_features=model_dim)
        self.mlp = GPT2FeedForwardNetwork(model_dim=model_dim, feedforward_dim=feedforward_dim)
        self.ln_2 = torch.nn.Linear(in_features=model_dim, out_features=model_dim)

    def load_weights(self, model_params: TransformerBlockParams):
        with torch.no_grad():
            self.ln_1.weight = torch.nn.Parameter(model_params.linear_1.weights)
            self.ln_1.bias = torch.nn.Parameter(model_params.linear_1.biases)
            self.ln_2.weight = torch.nn.Parameter(model_params.linear_2.weights)
            self.ln_2.bias = torch.nn.Parameter(model_params.linear_2.biases)
            self.mlp.load_weights(model_params.feed_foward)
            self.attn.load_weights(model_params.attention)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tensor = self.attn(input)
        tensor = self.ln_1(tensor)
        tensor = self.mlp(tensor)
        tensor = self.ln_2(tensor)
        return tensor