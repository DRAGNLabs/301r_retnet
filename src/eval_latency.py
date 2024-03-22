import json
import sys
import yaml
import torch
import time
import random
import subprocess

from argparse import ArgumentParser
from models import RetNetModelHF, TransformerModelHF
from pathlib import Path
from torchscale.architecture.config import RetNetConfig, DecoderConfig
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from typing import Optional, List
from utils import Struct
from generate import generate_text_length_n
from models import RetNetModel, TransformerModel

def eval_compute_time(config: Struct):
    test_sequence_lengths = [i * (config.seq_len // config.seq_len_frac) for i in range(1, config.seq_len_frac)]
    test_sequence_lengths.append(config.seq_len - 1)
    results = {}
    random_tokens = make_random_tokens(config, config.seq_len)
    max_len_generated = False

    for test_seq_len in test_sequence_lengths:
        iteration_results = {'single_token_times': [], 'n_token_times': []}
        input_tokens = random_tokens[:test_seq_len]

        print("Measuring single token generation time...")
        for _ in range(config.num_latency_trials):
            iteration_results["single_token_times"].append(eval_single_token_generation_time(config, input_tokens))

        print("Measuring n token generation time...")
        if test_seq_len >= config.seq_len - config.n_tokens_to_generate and not max_len_generated:
            for _ in range(config.num_latency_trials):
                iteration_results["n_token_times"].append(eval_n_tokens_generation_time(config, config.n_tokens_to_generate, input_tokens[:config.seq_len - config.n_tokens_to_generate]))
                max_len_generated = True
        else:
            for _ in range(config.num_latency_trials):
                iteration_results["n_token_times"].append(eval_n_tokens_generation_time(config, config.n_tokens_to_generate, input_tokens))
            results[test_seq_len] = iteration_results

    return results

# measure the time to generate a single token
def eval_single_token_generation_time(config: Struct, input_tokens: torch.Tensor):
    start = time.time()
    generate_text_length_n(config=config, n=1, input_tokens=input_tokens)
    end = time.time()
    generation_time = end - start
    return generation_time

# measure the time to generate n tokens
def eval_n_tokens_generation_time(config: Struct, n_tokens_to_generate: int, input_tokens: torch.Tensor):
    start = time.time()
    generation_time = generate_text_length_n(config=config, n=n_tokens_to_generate, input_tokens=input_tokens)
    end = time.time()
    generation_time = end - start
    return generation_time

def make_random_tokens(config: Struct, n_tokens: int):
    # generate n random tokens
    tokens = []
    for _ in range(n_tokens):
        tokens.append(random.randint(0, config.vocab_size - 1))
    return torch.tensor(tokens)

if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    eval_compute_time(config)