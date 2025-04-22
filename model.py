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
                 n_layer: int,
                 model_params: ModelParams):
        super(GPT2, self).__init__()

        # GPT-2 architecture:
        # We use a Transformer (Vaswani et al., 2017) based architecture for our LMs. The model largely follows the details of the OpenAI GPT model (Radford et al., 2018) with a few modifications. Layer normalization (Ba et al., 2016) was moved to the input of each sub-block, similar to a pre-activation residual network (He et al., 2016) and an additional layer normalization was added after the final selfattention block. A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers. The vocabulary is expanded to 50,257. We also increase the context size from 512 to 1024 tokens and a larger batchsize of 512 is used.

        # Input projection (encoder.json + model/wte)
        # Positional encoding (model/wpe)
        #
        # 12 layers of transformers:
        #     Multi-headed causal (masked) self-attention
        #     Layernorm 1
        #     FFN
        #     Layernorm 2
        #
        # final Layernorm
        #
        # Output projection (model/wte)

        self.vocab_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.position_embedding = torch.nn.Embedding(num_embeddings=context_length, embedding_dim=embedding_dim)

        self.transformer = torch.nn.ModuleList([
            transformer.GPT2TransformerDecoder(model_dim=embedding_dim,
                                               num_heads=n_head,
                                               feedforward_dim=3072) for _ in range(n_layer)])

        self.ln_f = torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.softmax = torch.nn.Softmax(dim=1)  # Each output is now [N, n_vocab] and we will argmax for the next vocab.

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
