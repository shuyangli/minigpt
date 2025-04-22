import dataclasses

@dataclasses.dataclass
class Hyperparams:
   n_vocab: int
   n_ctx: int
   n_embd: int
   n_head: int
   n_layer: int

class LinearParams:
    def __init__(self):
        self.weights = None
        self.biases = None

    def update_weights(self, name, value):
        if name[0] == "w" or name[0] == "g":
            self.weights = value
        elif name[0] == "b":
            self.biases = value
        else:
            raise ValueError(f"LinearParams received unexpected name {name} of shape {value.shape}")

class AttentionParams:
    def __init__(self):
        self.cross_attention = LinearParams()
        self.out_projection = LinearParams()

    def update_weights(self, name, value):
        if name[0] == "c_attn":
            self.cross_attention.update_weights(name[1:], value)
        elif name[0] == "c_proj":
            self.out_projection.update_weights(name[1:], value)
        else:
            raise ValueError(f"AttentionParams received unexpected name {name} of shape {value.shape}")

class FeedForwardParams:
    def __init__(self):
        self.up_projection = LinearParams()
        self.down_projection = LinearParams()

    def update_weights(self, name, value):
        if name[0] == "c_fc":
            self.up_projection.update_weights(name[1:], value)
        elif name[0] == "c_proj":
            self.down_projection.update_weights(name[1:], value)
        else:
            raise ValueError(f"FeedForwardParams received unexpected name {name} of shape {value.shape}")

class TransformerBlockParams:
    def __init__(self):
        self.attention = AttentionParams()
        self.linear_1 = LinearParams()
        self.feed_foward = FeedForwardParams()
        self.linear_2 = LinearParams()

    def update_weights(self, name, value):
        if name[0] == "attn":
            self.attention.update_weights(name[1:], value)
        elif name[0] == "ln_1":
            self.linear_1.update_weights(name[1:], value)
        elif name[0] == "ln_2":
            self.linear_2.update_weights(name[1:], value)
        elif name[0] == "mlp":
            self.feed_foward.update_weights(name[1:], value)
        else:
            raise ValueError(f"TransformerBlockParams received unexpected name {name} of shape {value.shape}")

class ModelParams:
    def __init__(self, num_layers: int):
        self.positional_embeddings = None
        self.token_embeddings = None
        self.transformers = [TransformerBlockParams() for i in range(num_layers)]
        self.linear = LinearParams()

    def update_weights(self, name, value):
        if name[0] == "ln_f":
            self.linear.update_weights(name[1:], value)
        elif name[0] == "wpe":
            self.positional_embeddings = value
        elif name[0] == "wte":
            self.token_embeddings = value
        elif name[0][0] == "h":
            layer = int(name[0][1:])
            self.transformers[layer].update_weights(name[1:], value)
        else:
            raise ValueError(f"Model received unexpected name {name} of shape {value.shape}")
