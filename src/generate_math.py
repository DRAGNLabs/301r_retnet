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
import pandas as pd

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
    model, device = prepare_model_device(config)

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
        generation_length=config.gen_len,
        n_shot=config.nshot)

    if config.csv_path is not None:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = config.csv_path[:-4] + f"_{config.nshot}_shot.csv"
        with open(file=filename, mode='w', newline='') as outf:
            writer = csv.writer(outf)
            for y, y_hat in generated_strings:
                pred_result = True if y in y_hat else False  # See if the target is in the prediction
                writer.writerow([current_time, config.model_type, y_hat, y, pred_result])

def generate_text(
        model: nn.Module,
        tokenizer: Tokenizer,
        generation_path: str,
        device: torch.device,
        seq_len: int,
        generation_length: int=100,
        n_shot: int=0) -> list[str]:
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
    n_prompts = 100

    # Keep track of fully generated token indices lists
    generated_token_idx_list = []

    # Create list of strings from all lines in start_string_list path
    df = pd.read_csv(generation_path, names=["data"], nrows=1000)

    prompts = df.data.tolist()
    # Randomly pick n_prompts lines from prompts
    prompts = random.sample(prompts, n_prompts * n_shot)

    # Split prompts into n_prompt lists of len n_shot
    prompts = [prompts[i:i+n_shot] for i in range(0, len(prompts), n_shot)]  # split prompts into list of N-shot-length sub lists
    
    full_prompts = []
    full_prompt_targets = []
    # Append each prompt list together into a main prompt list
    for prompt_list in prompts:
        n_shots = prompt_list[:n_shot-1]  # Nshot examples
        final_prompt = prompt_list[-1]  # test examples
        final_prompt_source = final_prompt[:final_prompt.index("=")+1]  # cut answer out of final example
        # Join together into single string
        full_prompt = "\n".join(n_shots) + "\n" + final_prompt_source
        full_prompt_targets.append("\n".join(prompt_list))
        full_prompts.append(full_prompt)

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
            len_input = len(input_token_idxs)

            # Store generated sequence token indices
            generated_token_idxs = input_token_idxs

            # Get tensor with token indices
            input_tensor = torch.tensor(input_token_idxs, dtype=torch.long)

            # Add batch dimension and move to device
            input_tensor = input_tensor.unsqueeze(0).to(device)
            print(f"\n\nStarting on item {input_idx}")

            # Keep generating until padding or reached generation length
            while generated_token_idxs[-1] != tokenizer.pad_token_id \
                    and len(generated_token_idxs) < generation_length+len_input:
                
                print(f"Generated {len(generated_token_idxs)} of {len_input+generation_length} tokens (includes prompt).")
                # Make sure input_tensor isn't longer than sequence length
                input_tensor = input_tensor[: , max(0, input_tensor.shape[-1] - seq_len):]

                # Get model predictions
                predictions = model(input_tensor)

                # Apply softmax to predictions
                predictions = torch.nn.functional.softmax(predictions, dim=-1)
                # print('\nSoftmaxed predictions:', predictions)


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
    
    return zip(decoded_predictions, full_prompt_targets)

if __name__ == "__main__":
    print('Starting')
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    generate(config)
    