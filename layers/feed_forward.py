import torch
from params.model_params import FeedForwardParams

class GPT2FeedForwardNetwork(torch.nn.Module):
    def __init__(self, model_dim: int, feedforward_dim: int):
        super(GPT2FeedForwardNetwork, self).__init__()

        # Up projection
        self.c_fc = torch.nn.Linear(in_features=model_dim, out_features=feedforward_dim)
        # Gelu
        self.gelu = torch.nn.GELU()
        # Down projection
        self.c_proj = torch.nn.Linear(in_features=feedforward_dim, out_features=model_dim)

    def load_weights(self, model_params: FeedForwardParams):
        with torch.no_grad():
            self.c_fc.weight = torch.nn.Parameter(model_params.up_projection.weights)
            self.c_fc.bias = torch.nn.Parameter(model_params.up_projection.biases)
            self.c_proj.weight = torch.nn.Parameter(model_params.down_projection.weights)
            self.c_proj.bias = torch.nn.Parameter(model_params.down_projection.biases)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.c_fc(input)
        output = self.gelu(output)
        output = self.c_proj(output)
        return output