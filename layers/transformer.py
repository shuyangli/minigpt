import torch
from .multi_headed_attention import MultiHeadedAttention
from .feed_forward import GPT2FeedForwardNetwork
from params.model_params import TransformerBlockParams

class GPT2TransformerDecoder(torch.nn.Module):
    def __init__(self, model_dim: int, num_heads: int, feedforward_dim: int):
        super(GPT2TransformerDecoder, self).__init__()

        self.layernorm_1 = torch.nn.LayerNorm(model_dim)
        self.attn = MultiHeadedAttention(embed_dim=model_dim, num_heads=num_heads)
        self.layernorm_2 = torch.nn.LayerNorm(model_dim)
        self.mlp = GPT2FeedForwardNetwork(model_dim=model_dim, feedforward_dim=feedforward_dim)

    def load_weights(self, model_params: TransformerBlockParams):
        with torch.no_grad():
            self.model_params = model_params
            self.layernorm_1.weight = torch.nn.Parameter(model_params.ln_1.gamma)
            self.layernorm_1.bias = torch.nn.Parameter(model_params.ln_1.beta)
            self.layernorm_2.weight = torch.nn.Parameter(model_params.ln_2.gamma)
            self.layernorm_2.bias = torch.nn.Parameter(model_params.ln_2.beta)

            self.mlp.load_weights(model_params.feed_foward)
            self.attn.load_weights(model_params.attention)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tensor_norm = self.layernorm_1(input)
        tensor = input + self.attn(tensor_norm)
        tensor_norm = self.layernorm_2(tensor)
        tensor = tensor + self.mlp(tensor_norm)
        return tensor