from models import RetNetModel
import torch
from torch import Tensor
import torch.nn as nn

import yaml
import sys
from utils import Struct, generate_text
import inference_test


def generate_specific_text(config):

    # Generate text from the model
    print("\nGenerating text...")
    input_starting_strings = [
        "<pad>",
        "= valkyria",
        "= = reception ="]


    if config.model_type.lower() == "retnet":
        model = RetNetModel(config)

    model.load_state_dict(torch.load(config.checkpoints_path))

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_path)

    device = torch.device(config.device)

    generated_strings = generate_text(
        model=model,
        tokenizer=tokenizer,
        start_string_list=input_starting_strings,
        device=device,
        seq_len=config.seq_len,
        generation_length=100)

    print("Generated strings:")
    for idx, string in enumerate(generated_strings):
        print(f"{idx+1}: {string}\n")

if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    generate_specific_text(config)
   