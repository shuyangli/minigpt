import argparse
import json
import tensorflow as tf
import torch

from layers import bpe
from pathlib import Path
from model import GPT2
from params.model_params import Hyperparams, ModelParams


def generate(encoder: bpe.BytePairEncoder, model: GPT2, input: list[str], num_tokens: int) -> list[str]:
    # Need to pad this appropriately for multiple inputs
    tokens = [encoder.encode(prompt) for prompt in input]
    tensor = torch.tensor(tokens, dtype=torch.int64)

    for _ in range(num_tokens):
        probabilities = model(tensor)
        next_token = torch.argmax(probabilities, dim=-1, keepdim=True)
        tensor = torch.cat((tensor, next_token), dim=-1)

    outputs = [encoder.decode(output) for output in tensor.tolist()]
    return outputs

def load_weights_and_hyperparams(model_path: Path) -> tuple[Hyperparams, ModelParams]:
    hyperparam_path = model_path / Path("hparams.json")
    with open(hyperparam_path, "r") as hyperparams_file:
        hyperparams = Hyperparams(**json.loads(hyperparams_file.read()))

    checkpoint = tf.train.latest_checkpoint("./gpt2-124M")
    model_params = ModelParams(hyperparams.n_layer)

    for name, _ in tf.train.list_variables(checkpoint):
        param = tf.train.load_variable(checkpoint, name)
        model_params.update_weights(name.split("/")[1:], param)

    return hyperparams, model_params


def main():
    parser = argparse.ArgumentParser(prog="minigpt2", description="Runs a mini GPT-2 model")
    parser.add_argument("prompt", nargs="+")
    parser.add_argument("-n", "--num_tokens")

    args = parser.parse_args()
    prompt = args.prompt
    num_tokens = int(args.num_tokens)

    model_path = Path("./gpt2-124M")
    hyperparams, model_params = load_weights_and_hyperparams(model_path)
    byte_pair_encoder = bpe.get_encoder(model_path)

    model = GPT2(hyperparams.n_vocab, hyperparams.n_embd, hyperparams.n_ctx, hyperparams.n_head, hyperparams.n_layer)
    model.load_weights(model_params)

    output = generate(byte_pair_encoder, model, prompt, num_tokens)

    print(output)

if __name__ == "__main__":
    main()
