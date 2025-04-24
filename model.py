import torch
from layers import transformer
from params.model_params import ModelParams

class GPT2(torch.nn.Module):
    """A minimal implementation of the GPT2 model"""
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 context_length: int,
                 n_head: int,
                 n_layer: int):
        super(GPT2, self).__init__()

        self.vocab_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.position_embedding = torch.nn.Embedding(num_embeddings=context_length, embedding_dim=embedding_dim)

        self.transformer = torch.nn.ModuleList([
            transformer.GPT2TransformerDecoder(model_dim=embedding_dim,
                                               num_heads=n_head,
                                               feedforward_dim=3072) for _ in range(n_layer)])
        # Each output is now [N, n_vocab] and we will argmax for the next vocab.
        self.layernorm = torch.nn.LayerNorm(embedding_dim)
        self.lm_head = torch.nn.Linear(embedding_dim, vocab_size, bias=False)

    def load_weights(self, model_params: ModelParams):
        with torch.no_grad():
            self.vocab_embedding.weight = torch.nn.Parameter(model_params.token_embeddings)
            self.position_embedding.weight = torch.nn.Parameter(model_params.positional_embeddings)
            for decoder, weights in zip(self.transformer, model_params.transformers):
                decoder.load_weights(weights)
            self.layernorm.weight = torch.nn.Parameter(model_params.linear.weights)
            self.layernorm.bias = torch.nn.Parameter(model_params.linear.biases)
            self.lm_head.weight = torch.nn.Parameter(model_params.token_embeddings)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input.shape

        token_embeddings = self.vocab_embedding(input)  # [N, n_dims]

        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input)  # [batch_size, seq_length]
        position_embeddings = self.position_embedding(position_ids)

        x = token_embeddings + position_embeddings      # [N, n_dims]

        for transformer_block in self.transformer:
            x = transformer_block(x)

        x = self.layernorm(x)

        # Unembed
        last_token = x[:, -1, :]
        logits = self.lm_head(last_token)
        return logits
