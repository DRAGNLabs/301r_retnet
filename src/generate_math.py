print('hello world')

import csv
import sys
import random
import torch
import yaml

from datetime import datetime
from models import LongNetModel, RetNetModel, TransformerModel
from tokenizers import Tokenizer
from torch import nn
from transformers import PreTrainedTokenizerFast
from typing import List
from utils import Struct

def prepare_model_device(config: Struct):
    """
    Args:
        config (Struct): A Struct object with all configuration fields.
    """
    # Create appropriate model type
    if config.model_type.lower() == "longnet":
        model = LongNetModel(config)
    elif config.model_type.lower() == "retnet":
        model = RetNetModel(config)
    elif config.model_type.lower() == "transformer":
        model = TransformerModel(config)
    else:
        raise ValueError(f"Model type '{config.model_type}' not supported!")

    # Load in pre-trained weights from checkpoint
    if config.checkpoint_path is None:
        raise ValueError(
            "To generate text, the 'checkpoint_path' value in the " + \
            "configuration file must be set")
    checkpoint = torch.load(config.checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    device = torch.device(config.device)
    return model, device

def generate(config: Struct):
    """
    Args:
        config (Struct): A Struct object with all configuration fields.
    """
    print('Preparing model devices...')
    #model, device = prepare_model_device(config)
    model=None
    device=None

    print("Loading tokenizer...")
    # Load pre-trained tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_path)

    # Get completed strings
    print("Generating text...")
    generated_strings = generate_text(
        model=model,
        tokenizer=tokenizer,
        generation_path=config.generation_path,
        device=device,
        seq_len=config.seq_len,
        generation_length=config.gen_len)

    print("Generated strings:")
    for idx, string in enumerate(generated_strings):
        print(f"{idx+1}: {string}\n")

    if config.csv_path is not None:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(config.csv_path, 'a', newline='') as outf:
            writer = csv.writer(outf)
            for string in generated_strings:
                writer.writerow([current_time, config.model_type, string])

def generate_text(
        model: nn.Module,
        tokenizer: Tokenizer,
        generation_path: str,
        device: torch.device,
        seq_len: int,
        generation_length: int=100) -> list[str]:
    """ Use model to generate text given beginning input
    Args:
        model (nn.Module): Model used to make predictions.
        tokenizer (Tokenizer): Tokenizer object to handle conversions with
            tokens and strings.
        generation_path (list[str]): List of strings that should be used as
            beginnings of generated strings (prompts to the model).
        device (torch.device): Device on which to run inference.
        seq_len (int): Context window/sequence length.
        generation_length (int): Total amount of tokens per generated sequence.

    Returns:
        A list of all the fully generated strings by the model.
    """
    n_prompts = 1
    n_shot = 6

    # Keep track of fully generated token indices lists
    generated_token_idx_list = []

    # Create list of strings from all lines in start_string_list path
    with open(generation_path, "r") as f:
        prompts = f.read().split("\n")[:10000]

    # Randomly pick n_prompts lines from prompts
    prompts = random.sample(prompts, n_prompts * n_shot)

    # Split prompts into n_shot lists of n_prompts
    prompts = [prompts[i:i+n_shot] for i in range(0, len(prompts), n_shot)]

    full_prompts = []
    full_prompt_targets = []
    # Append each prompt list together into a main prompt list
    for prompt_list in prompts:
        n_shots = prompt_list[:n_shot-1]
        final_prompt = prompt_list[-1]
        final_prompt_source = final_prompt[:final_prompt.index("=")+1]
        # Join together into single string
        full_prompt = prompts.append("\n".join(n_shots) + "\n" + final_prompt_source)
        full_prompt_targets.append("\n".join(prompt_list))
        full_prompts.append(full_prompt)

    print('full_prompts:', full_prompts)
    print('full_prompt_targets:', full_prompt_targets)

    # Convert initial start strings into tokenized sequences
    tokenized_start_list = tokenizer(
        full_prompts,
        padding=False,
        truncation=False,
        return_token_type_ids=False,
        return_attention_mask=False)["input_ids"]

    # Move model to device
    model = model.to(device)

    # No gradients needed
    model.eval()
    with torch.inference_mode():
        # Iterate over each start string
        for input_idx in range(len(tokenized_start_list)):
            # Retrieve string's tokenized version
            input_token_idxs = tokenized_start_list[input_idx]

            # Store generated sequence token indices
            generated_token_idxs = input_token_idxs

            # Get tensor with token indices
            input_tensor = torch.tensor(input_token_idxs, dtype=torch.long)

            # Add batch dimension and move to device
            input_tensor = input_tensor.unsqueeze(0).to(device)

            # Keep generating until padding or reached generation length
            while generated_token_idxs[-1] != tokenizer.pad_token_id \
                    and len(generated_token_idxs) < generation_length:
                # Make sure input_tensor isn't longer than sequence length
                input_tensor = input_tensor[
                    :,
                    max(0, input_tensor.shape[-1] - seq_len):]

                # Get model predictions
                predictions = model(input_tensor)

                # Apply softmax to predictions
                predictions = torch.functional.softmax(predictions, dim=-1)

                # Get the last predicted word
                predicted_id = predictions.argmax(dim=-1)[0, -1]

                # Add predicted word to input (used as next input sequence)
                input_tensor = torch.cat(
                    [input_tensor, predicted_id[None, None]],
                    dim=-1)

                # Store predicted token as part of generation
                generated_token_idxs.append(predicted_id.item())

            # Store fully generated sequence of token indices for start string
            generated_token_idx_list.append(generated_token_idxs)

    # Decode token indices lists to lists of strings and return
    decoded_predictions = tokenizer.batch_decode(generated_token_idx_list)

    return decoded_predictions

if __name__ == "__main__":
    print('Starting')
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    generate(config)
    