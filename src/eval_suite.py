import json
import lm_eval
from models import LongNetModelHF, RetNetModelHF, TransformerModelHF
from torchscale.architecture.config import DecoderConfig, LongNetConfig, RetNetConfig
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
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
    AutoConfig.register("longnet", LongNetConfig)
    AutoModel.register(LongNetConfig, LongNetModelHF)
    AutoModel.register(RetNetConfig, RetNetModelHF)
    AutoModel.register(DecoderConfig, TransformerModelHF)
    AutoModelForCausalLM.register(LongNetConfig, LongNetModelHF)
    AutoModelForCausalLM.register(RetNetConfig, RetNetModelHF)
    AutoModelForCausalLM.register(DecoderConfig, TransformerModelHF)

    task_manager = lm_eval.tasks.TaskManager()

    if config.tokenizer_path:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={config.model_path_dir}," + \
                f"tokenizer={config.tokenizer_path}",
            tasks=config.tasks,
            num_fewshot=config.nshot,
            device=config.device,
            task_manager=task_manager,
            )
    else:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={config.model_path_dir}",
            tasks=config.tasks,
            num_fewshot=0,
            device=config.device,
            task_manager=task_manager)

    print(results["results"])

    with open(config.results_out_path, "w") as f:
        json.dump(results["results"], f, indent=4)
