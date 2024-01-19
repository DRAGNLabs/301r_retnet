import json
import argparse
from pathlib import Path
import lm_eval
from typing import Optional, List
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
from hugging_face_model import RetNetModel, TransformerModel
from torchscale.architecture.config import RetNetConfig, DecoderConfig



def run_eval(model_path_dir: str, tokenizer_path_dir: Optional[str] = None, tasks: List[str] = ['hellaswag'], device: str = 'cuda:0', results_out_path: Optional[str] = None):
    """
    Run evaluation on all tasks
    :param model_path_dir: path to model directory
    :param tokenizer_path_dir: path to tokenizer directory (Required only if tokenizer is not in model directory)
    :return: None
    """
    AutoConfig.register('retnet', RetNetConfig)
    AutoConfig.register('custom_transformer', DecoderConfig)
    AutoModel.register(RetNetConfig, RetNetModel)
    AutoModel.register(DecoderConfig, TransformerModel)
    AutoModelForCausalLM.register(RetNetConfig, RetNetModel)
    AutoModelForCausalLM.register(DecoderConfig, TransformerModel)



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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path_dir', type=str, required=True, help='Path to model directory')
    parser.add_argument('--tokenizer_path_dir', type=str, default=None, help='Path to tokenizer directory')
    parser.add_argument('--tasks', type=str, nargs='+', default=['hellaswag'], help='Tasks to evaluate on')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')
    parser.add_argument('--results_out_path', type=str, default=None, help='Path to save results')
    args = parser.parse_args()

    run_eval(args.model_path_dir, args.tokenizer_path_dir, args.tasks, args.device, args.results_out_path)
