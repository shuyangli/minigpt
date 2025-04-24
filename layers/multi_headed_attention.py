import torch
from params.model_params import AttentionParams

class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_projection = torch.nn.Linear(embed_dim, embed_dim * 3)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.output_projection = torch.nn.Linear(embed_dim, embed_dim)

    def load_weights(self, model_params: AttentionParams):
        with torch.no_grad():
            self.qkv_projection.weight = torch.nn.Parameter(model_params.c_attn.weights.squeeze(dim=0).T)
            self.qkv_projection.bias = torch.nn.Parameter(model_params.c_attn.biases)
            self.output_projection.weight = torch.nn.Parameter(model_params.out_projection.weights.squeeze(dim=0).T)
            self.output_projection.bias = torch.nn.Parameter(model_params.out_projection.biases)

    def forward(self, input: torch.Tensor):
        batch_size, seq_length, embedding_dims = input.shape
        # [batch_size, seq_length, (query, key, value) joined]
        qkv = self.qkv_projection(input)

        # Split qkv by reshaping
        # [batch_size, seq_length, 3, num_heads, head_dim]
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        # [3 (q/k/v), batch_size, num_head, seq_length, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # [batch_size, num_head, seq_length, head_dim]
        query, key, value = qkv[0], qkv[1], qkv[2]

        scaling_factor = self.head_dim ** 0.5

        # Final two dimensions are matmul dimensions;
        # previous dims are batch dims and are broadcasted.
        # [batch_size, num_head, seq_length, seq_length]
        output = query @ key.transpose(-2, -1) / scaling_factor

        # Causal mask for each batch / head
        mask = torch.tril(torch.ones((seq_length, seq_length))).reshape((1, 1, seq_length, seq_length))
        output = output.masked_fill(mask == 0, float("-inf"))

        output = self.softmax(output)

        # [batch_size, num_head, seq_length, head_dim]
        output = output @ value

        # Permute them back to [batch_size, seq_length, num_head, head_dim]
        output = output.permute((0, 2, 1, 3))

        # The heads are now joined via reshape
        output = output.reshape((batch_size, seq_length, self.embed_dim))
        output = self.output_projection(output)
        return output