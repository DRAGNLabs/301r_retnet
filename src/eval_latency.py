import json
import random
import subprocess
import sys
import time
import torch
import yaml

from argparse import ArgumentParser
from generate import generate_text_length_n
from models import RetNetModel, TransformerModel, RetNetModelHF, TransformerModelHF
from pathlib import Path
from torchscale.architecture.config import DecoderConfig, RetNetConfig
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PreTrainedTokenizerFast
from typing import List, Optional
from utils import Struct

def eval_latency(config: Struct):
    """
    Args:
        config (Struct): A Struct object with all configuration fields.

    Format of results will be:
        {
            test_seq_len: {
                'single_token_times': {
                    [trial1, trial2,...]
                },

                'n_token_times': {
                    [trial1, trial2,...]
                }
            }
        }
    """
    results = {}

    test_sequence_lengths = config.test_sequence_lengths.sort()
    if len(test_sequence_lengths) == 0:
        test_sequence_lengths = [config.seq_len]

    num_latency_trials = config.num_latency_trials
    if not num_latency_trials:
        num_latency_trials = 5

    n_tokens_to_generate = config.n_tokens_to_generate
    if not n_tokens_to_generate:
        n_tokens_to_generate = 10

    # Generate a list of pretokenized data 
    random_tokens = [random.randint(0, config.vocab_size - 1) for _ in range(test_sequence_lengths[-1])]

    # The first couple token generation take longer than all the following generations
    # Generating these few tokens should alleviate that behavior
    generate_text_length_n(config=config, n=1, input_tokens=random_tokens[0:1])
    generate_text_length_n(config=config, n=1, input_tokens=random_tokens[0:1])

    # Evaluate token generation at each prescribed sequence length
    for test_seq_len in test_sequence_lengths:
        iteration_results = {'single_token_times': [], 'n_token_times': []}
        input_tokens = random_tokens[:test_seq_len]

        # Measure the time to generate a single token
        print("Measuring single token generation time...")
        for _ in range(num_latency_trials):
            generation_time = generate_text_length_n(config=config, n=1, input_tokens=input_tokens)
            iteration_results["single_token_times"].append(generation_time)

        # Measure the tme to generate n tokens
        print(f"Measuring {test_seq_len} token generation time...")
        for _ in range(num_latency_trials):
            generation_time = generate_text_length_n(config=config, n=n_tokens_to_generate, input_tokens=input_tokens)
            iteration_results["n_token_times"].append(generation_time)

        results[test_seq_len] = iteration_results

    print(results)

    with open(config.results_out_path, "w") as f:
        json.dump(results, f, indent=4)
