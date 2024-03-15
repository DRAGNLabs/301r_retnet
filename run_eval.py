import json
import lm_eval
import sys
import yaml

from argparse import ArgumentParser
from models import RetNetModelHF, TransformerModelHF
from pathlib import Path
from torchscale.architecture.config import RetNetConfig, DecoderConfig
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
from typing import Optional, List
from utils import Struct

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
    AutoConfig.register("performer", PerformerConfig)
    AutoModel.register(RetNetConfig, RetNetModelHF)
    AutoModel.register(DecoderConfig, TransformerModelHF)
    AutoModel.register(PerformerConfig, PerformerModelHF)
    AutoModelForCausalLM.register(RetNetConfig, RetNetModelHF)
    AutoModelForCausalLM.register(DecoderConfig, TransformerModelHF)
    AutoModelForCausalLM.register(PerformerConfig, PerformerModelHF)

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


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    run_eval(config)
