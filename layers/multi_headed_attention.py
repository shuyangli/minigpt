import torch
from params.model_params import AttentionParams

class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadedAttention, self).__init__()
        self.qkv_projection = torch.nn.Linear(embed_dim, embed_dim * 3)
        self.softmax = torch.nn.Softmax(embed_dim)
        self.output_projection = torch.nn.Linear(embed_dim * 3, embed_dim)
        self.num_heads = num_heads

    def load_weights(self, model_params: AttentionParams):
        with torch.no_grad():
            self.qkv_projection.weight = torch.nn.Parameter(model_params.c_attn.weights)
            self.qkv_projection.bias = torch.nn.Parameter(model_params.c_attn.biases)
            self.output_projection.weight = torch.nn.Parameter(model_params.out_projection.weights)
            self.output_projection.bias = torch.nn.Parameter(model_params.out_projection.biases)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        output = query @ key
        # TODO: Mask
        output = self.softmax(output)
        output = output @ value
        return output

    def forward(self, input: torch.Tensor):
        print(f"MultiHeadedAttention {input.shape=}")
        projected = self.qkv_projection(input)
        print(f"MultiHeadedAttention {projected.shape=}")

        query, key, value = torch.split(projected, 3, dim=1)

        query_heads = torch.split(query, self.num_heads, dim=1)
        key_heads = torch.split(key, self.num_heads, dim=1)
        value_heads = torch.split(value, self.num_heads, dim=1)
        output_heads = [self._attention(query, key, value) for query, key, value in zip(query_heads, key_heads, value_heads)]

        output = torch.hstack(output_heads)
        output = self.output_projection(output)

        return output