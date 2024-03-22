import json
import lm_eval
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

def run_eval(config: Struct):
    """ Run evaluation on all tasks (benchmarks).
    Args:
        config (Struct): A Struct object with all configuration fields.
    """
    if config.model_path_dir is None:
        raise ValueError("model_path_dir requires a path, but none was given.")
    if config.results_out_path is None:
        raise ValueError("results_out_path requires a path, but none was " + \
            "given.")

    AutoConfig.register("retnet", RetNetConfig)
    AutoConfig.register("custom_transformer", DecoderConfig)
    AutoModel.register(RetNetConfig, RetNetModelHF)
    AutoModel.register(DecoderConfig, TransformerModelHF)
    AutoModelForCausalLM.register(RetNetConfig, RetNetModelHF)
    AutoModelForCausalLM.register(DecoderConfig, TransformerModelHF)

    lm_eval.tasks.initialize_tasks()
    if config.tokenizer_path:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={config.model_path_dir}," + \
                f"tokenizer={config.tokenizer_path}",
            tasks=config.tasks,
            num_fewshot=0,
            device=config.device)
    else:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={config.model_path_dir}",
            tasks=config.tasks,
            num_fewshot=0,
            device=config.device)

    print(results["results"])

    with open(config.results_out_path, "w") as f:
        json.dump(results["results"], f, indent=4)

def eval_compute_time(config: Struct, single_token_num_trials: int, n_token_num_trials: int, n_tokens: int, seq_len_split: int):
    # find best, worst, and average times
    test_sequence_lengths = [n_tokens // seq_len_split] * seq_len_split
    single_token_times = []
    n_token_times = []

    for seq_len in test_sequence_lengths:
        random_tokens = make_random_tokens(config, seq_len)
        print("Measuring single token generation time...")
        for _ in range(single_token_num_trials):
            single_token_times.append(eval_single_token_time(config))

        print("Measuring n token generation time...")
        for _ in range(n_token_num_trials):
            n_token_times.append(eval_n_tokens_time(config, n_tokens))

        return single_token_times, n_token_times

def eval_single_token_time(config: Struct):
    # measure the time to generate a single token
    start = time.time()
    generate_text_length_n(config, 1)
    end = time.time()
    generation_time = end - start
    return generation_time

def eval_n_tokens_time(config: Struct, n_tokens: int):
    # measure the time to generate n tokens
    start = time.time()
    generation_time = generate_text_length_n(config, n_tokens)
    end = time.time()
    generation_time = end - start
    return generation_time

def make_random_tokens(config: Struct, n_tokens: int):
    # generate n random tokens
    tokens = []
    for _ in range(n_tokens):
        tokens.append(random.randint(0, config.vocab_size - 1))
    return tokens

if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    run_eval(config)
