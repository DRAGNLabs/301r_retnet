import csv
import sys
import random
import torch
import yaml

from datetime import datetime
from models import LongNetModel, RetNetModel, TransformerModel
from tokenizers import Tokenizer
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from typing import List
from utils import Struct, load_vocab_as_dict
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
    device = torch.device(config.device)
    checkpoint = torch.load(config.checkpoint_path, map_location = device)
    model.load_state_dict(checkpoint["state_dict"])

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
        filename = config.csv_path[:-4] + f"_{config.nshot}_shot_{config.model_type}.csv"
        with open(file=filename, mode='w', newline='') as outf:
            writer = csv.writer(outf)
            for prompt, y, y_hat in generated_strings: #, p_correct
                pred_result = True if y in y_hat else False  # See if the target is in the prediction
                writer.writerow([current_time, config.model_type, prompt, y, y_hat, pred_result])#, p_correct])
        print("\nDone!\n", flush=True)

def generate_text(
        model: nn.Module,
        tokenizer: Tokenizer,
        generation_path: str,
        device: torch.device,
        seq_len: int,
        generation_length: int=100,
        n_shot: int=0) -> List[str]:
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
    n_prompts = 2
    T = 1  # temperature

    # Keep track of fully generated token indices lists
    generated_token_idx_list = []

    # Create list of strings from all lines in start_string_list path
    rows:list[list] = []
    with open(generation_path, 'r') as infile:
        csv_reader = csv.reader(infile) 
        header = next(csv_reader)
        for row in csv_reader:
            rows.append(row[0])

    # Randomly pick n_prompts lines from prompts
    try:
        rows = random.sample(rows, n_prompts * (n_shot+1))
    except ValueError:
        random.shuffle(rows)
        print(f"WARNING: Random sample was higher than sampling pool;\
defaulting to shuffling entire pool and returning list of len {len(rows)}.\n")

    # Split prompts into n_prompt lists of len n_shot
    n_shot_rows = [rows[i:i+n_shot+1] for i in range(0, len(rows), n_shot+1)]  # split prompts into list of N-shot-length sub lists

    n_shot_prompts = []
    ground_truths = []
    # Append each prompt list together into a main prompt list
    for n_shot_row in n_shot_rows:
        final_prompt = n_shot_row[-1]  # LM's completion task
        prompt_split = final_prompt.index("=")+2
        LMs_prompt = final_prompt[:prompt_split]  # cut answer out of LM's prompt; leave the space following the '='.
        ground_truths.append(final_prompt[prompt_split:])  # get the actual answer that the LM needs to predict.
        complete_prompt = "\n".join(n_shot_row[:-1]+[LMs_prompt])
        n_shot_prompts.append(complete_prompt) 

    target_tokens = []

    tokenizer_path = config.tokenizer_path + "/tokenizer.json"
    vocab = load_vocab_as_dict(tokenizer_path)
    for target_word in ground_truths:
        token_ids = [v for k, v in vocab.items() if target_word.startswith(k)]
        target_tokens.append(token_ids)

    # Convert initial start strings into tokenized sequences
    prompts_in_tokens: List[List[int]] = tokenizer(
        n_shot_prompts,
        padding=False,
        truncation=False,
        return_token_type_ids=False,
        return_attention_mask=False)["input_ids"]  # TODO: Look into this field vs other fields; maybe we want more out of it.

    p_corrects = []

#    Move model to device
    model = model.to(device)

    # No gradients needed
    model.eval()
    with torch.inference_mode():
        # Iterate over each start string
        for i in tqdm(range(len(prompts_in_tokens))):
            # Retrieve string's tokenized version
            input_token_idxs = prompts_in_tokens[i]

            # Store generated sequence token indices
            generated_token_idxs = [float('-inf')]  # Instantiate listAvoid index error in first pass of while loop

            # Get tensor with token indices
            input_tensor = torch.tensor(input_token_idxs, dtype=torch.long)
            print(f"\n\nInput tensor size: {input_tensor.shape}\n\n")
            # Add batch dimension and move to device
            input_tensor = input_tensor.unsqueeze(0).to(device)
            print(f"\n\nStarting on item {i}", flush=True)

            num_generated_tokens = 0

            # Keep generating until padding or reached generation length
            while generated_token_idxs[-1] != tokenizer.pad_token_id \
                    and len(generated_token_idxs) < generation_length:

                if generated_token_idxs[-1] == float('-inf'):  # get rid of error handling non-token.
                    generated_token_idxs.pop()

                print(f"Generated {len(generated_token_idxs)} of {generation_length} token.")
                # Make sure input_tensor isn't longer than sequence length
                input_tensor = input_tensor[: , max(0, input_tensor.shape[-1] - seq_len):]

                # Get model predictions
                predictions = model(input_tensor)

                # Apply softmax to predictions
                predictions = torch.nn.functional.softmax(predictions / T, dim=-1)
                # Get the last predicted word
                predicted_id = predictions.argmax(dim=-1)[0, -1]
                print(predictions, flush=True)

                if not num_generated_tokens:
                    # Sum probabilities of all correct next tokens
                    print(predictions.size(), flush=True)
                    p_correct = 0 # Sum the probs of all correct next tokens
                    for token_idx in target_tokens[i]:
                        p_correct += predictions[0][-1][token_idx]  # This assumes that predictions is a tensor len 50k (ie vocab size) where token 1 is in position 0 etc.
                    p_corrects.append(p_correct)

                # Add predicted word to input (used as next input sequence)
                input_tensor = torch.cat([input_tensor, predicted_id[None, None]], dim=-1)

                # Store predicted token as part of generation
                generated_token_idxs.append(predicted_id.item())
                
                # Increment number of generated tokens
                num_generated_tokens += 1

            # Store fully generated sequence of token indices for start string
            generated_token_idx_list.append(generated_token_idxs)

    # Decode token indices lists to lists of strings and return
    decoded_predictions = tokenizer.batch_decode(generated_token_idx_list)
    
    return zip(n_shot_prompts, ground_truths, decoded_predictions)#, p_correct)

if __name__ == "__main__":
    print('Starting')
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    generate(config)
