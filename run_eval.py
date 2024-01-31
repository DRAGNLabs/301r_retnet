import json
import lm_eval

from argparse import ArgumentParser
from hugging_face_model import RetNetModelHF, TransformerModelHF
from pathlib import Path
from torchscale.architecture.config import RetNetConfig, DecoderConfig
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
from typing import Optional, List

def run_eval(
        model_path_dir: str,
        device: str = 'cuda:0',
        results_out_path: Optional[str] = None,
        tasks: List[str] = ['hellaswag'],
        tokenizer_path_dir: Optional[str] = None):
    """
    Run evaluation on all tasks
    :param model_path_dir: path to model directory
    :param tokenizer_path_dir: path to tokenizer directory (Required only if tokenizer is not in model directory)
    :return: None
    """
    AutoConfig.register('retnet', RetNetConfig)
    AutoConfig.register('custom_transformer', DecoderConfig)
    AutoModel.register(RetNetConfig, RetNetModelHF)
    AutoModel.register(DecoderConfig, TransformerModelHF)
    AutoModelForCausalLM.register(RetNetConfig, RetNetModelHF)
    AutoModelForCausalLM.register(DecoderConfig, TransformerModelHF)



    lm_eval.tasks.initialize_tasks()
    if tokenizer_path_dir:
        results = lm_eval.simple_evaluate(
            model='hf',
            model_args=f"pretrained={model_path_dir},tokenizer={tokenizer_path_dir}",
            tasks=tasks,
            num_fewshot=0,
            device=device,
        )
    else:
        results = lm_eval.simple_evaluate(
            model='hf',
            model_args=f"pretrained={model_path_dir}",
            tasks=tasks,
            num_fewshot=0,
            device=device,
        )

    print(results['results'])
    model_name = Path(model_path_dir).name
    if not results_out_path:
        print(f"No results path specified, saving to {model_name}_results.json")
        results_out_path = f"{model_name}_results.json"

    with open(results_out_path, 'w') as f:
        json.dump(results['results'], f, indent=4)


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')
    parser.add_argument('--model-path-dir', type=str, required=True, help='Path to model directory')
    parser.add_argument('--results-out-path', type=str, default=None, help='Path to save results')
    parser.add_argument('--tasks', type=str, nargs='+', default=['hellaswag'], help='Tasks to evaluate on')
    parser.add_argument('--tokenizer-path-dir', type=str, default=None, help='Path to tokenizer directory')

    args = parser.parse_args()
    run_eval(**vars(args))