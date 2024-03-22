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
    # find best, worst, and average times
    test_sequence_lengths = [config.n_tokens_to_generate // config.seq_len_frac] * config.seq_len_frac
    results = {}

    for seq_len in test_sequence_lengths:
        iteration_results = {'single_token_times': [], 'n_token_times': []}
        random_tokens = make_random_tokens(config, seq_len)
        print("Measuring single token generation time...")
        for _ in range(config.num_latency_trials):
            iteration_results["single_token_times"].append(eval_single_token_generation_time(config, random_tokens))

        print("Measuring n token generation time...")
        for _ in range(config.num_latency_trials):
            iteration_results["n_token_times"].append(eval_n_tokens_generation_time(config, config.n_tokens_to_generate, random_tokens))
        results[seq_len] = iteration_results

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
    # args = sys.argv
    # config_path = args[1]


    with open('configs/user_configs/eval_latency_test.yaml', "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    eval_compute_time(config)