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

        self.ln_f = torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.softmax = torch.nn.Softmax(dim=1)  # Each output is now [N, n_vocab] and we will argmax for the next vocab.

    def load_weights(self, model_params: ModelParams):
        with torch.no_grad():
            self.vocab_embedding.weight = torch.nn.Parameter(model_params.token_embeddings)
            self.position_embedding.weight = torch.nn.Parameter(model_params.positional_embeddings)
            self.ln_f.weight = torch.nn.Parameter(model_params.linear.weights)
            self.ln_f.bias = torch.nn.Parameter(model_params.linear.biases)
            for decoder, weights in zip(self.transformer, model_params.transformers):
                decoder.load_weights(weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        print(f"{input.shape=}")

        token_embeddings = self.vocab_embedding(input)  # [N, n_dims]
        position_embeddings = torch.arange(start=0, end=input.shape[0]) # [N, n_dims]
        position_embeddings = self.position_embedding(position_embeddings)

        x = token_embeddings + position_embeddings      # [N, n_dims]
        for transformer_block in self.transformer:
            x = transformer_block(x)
        x = self.ln_f(x)
        x = self.softmax(x)
        return x
