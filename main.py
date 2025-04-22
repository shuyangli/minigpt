import torch
import tensorflow as tf
import json

from layers import bpe
from pathlib import Path
from model import GPT2
from params.model_params import Hyperparams, ModelParams


def generate(encoder: bpe.BytePairEncoder, model: GPT2, input: str, num_tokens: int) -> str:
    tokens = encoder.encode(input)
    tensor = torch.tensor(tokens, dtype=torch.int64)
    print(tensor)

    for _ in range(num_tokens):
        probabilities = model(tensor)
        next_token = torch.argmax(probabilities, dim=1)
        print(f"next_token: {encoder.decode(next_token.tolist())}")
        tensor = torch.cat((tensor, next_token), dim=1)

    return encoder.decode(tensor.tolist())

def load_weights_and_hyperparams(model_path: Path) -> tuple[Hyperparams, ModelParams]:
    hyperparam_path = model_path / Path("hparams.json")
    with open(hyperparam_path, "r") as hyperparams_file:
        hyperparams = Hyperparams(**json.loads(hyperparams_file.read()))
    print(hyperparams)

    checkpoint = tf.train.latest_checkpoint("./gpt2-124M")
    model_params = ModelParams(hyperparams.n_layer)

    for name, _ in tf.train.list_variables(checkpoint):
        param = tf.train.load_variable(checkpoint, name)
        model_params.update_weights(name.split("/")[1:], param)

    return hyperparams, model_params


def main():
    model_path = Path("./gpt2-124M")
    hyperparams, model_params = load_weights_and_hyperparams(model_path)

    model = GPT2(hyperparams.n_vocab, hyperparams.n_embd, hyperparams.n_ctx, hyperparams.n_head, hyperparams.n_layer, model_params)

    byte_pair_encoder = bpe.get_encoder(model_path)
    print(byte_pair_encoder.encode("this is a test"))

    output = generate(byte_pair_encoder, model, "this is a test", 8)
    print(output)

if __name__ == "__main__":
    main()
