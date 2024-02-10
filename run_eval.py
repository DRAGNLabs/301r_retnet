import json
import lm_eval
import sys
import yaml

from argparse import ArgumentParser
from hugging_face_model import RetNetModelHF, TransformerModelHF
from pathlib import Path
from torchscale.architecture.config import RetNetConfig, DecoderConfig
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
from typing import Optional, List
from utils import Struct

def run_eval(config):
    """
    Run evaluation on all tasks
    :return: None
    """
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
            model_args=f"pretrained={config.model_path_dir},tokenizer={config.tokenizer_path}",
            tasks=config.tasks,
            num_fewshot=0,
            device=config.device,
        )
    else:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={config.model_path_dir}",
            tasks=config.tasks,
            num_fewshot=0,
            device=config.device,
        )

    print(results["results"])
    model_name = Path(config.model_path_dir).name
    results_out_path = config.results_out_path

    if not results_out_path:
        print(f"No results path specified, saving to {model_name}_results.json")
        results_out_path = f"{model_name}_results.json"

    with open(results_out_path, "w") as f:
        json.dump(results["results"], f, indent=4)

if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    run_eval(config)
