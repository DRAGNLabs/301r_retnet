import time
import torch
import torch.nn.functional as F

from tokenizers import Tokenizer
from torch import nn

def generate_text(
        model: nn.Module,
        tokenizer: Tokenizer,
        start_string_list: list[str],
        device: torch.device,
        seq_len: int,
        generation_length: int=100) -> list[str]:
    """ Use model to generate text given beginning input
    Args:
        model (nn.Module): Model used to make predictions.
        tokenizer (Tokenizer): Tokenizer object to handle conversions with
            tokens and strings.
        start_string_list (list[str]): List of strings that should be used as
            beginnings of generated strings (prompts to the model).
        device (torch.device): Device on which to run inference.
        seq_len (int): Context window/sequence length.
        generation_length (int): Total amount of tokens per generated sequence.

    Returns:
        A list of all the fully generated strings by the model.
    """
    # Keep track of fully generated token indices lists
    generated_token_idx_list = []

    # Convert initial start strings into tokenized sequences
    tokenized_start_list = tokenizer(
        start_string_list,
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
                predictions = F.softmax(predictions, dim=-1)

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
    return tokenizer.batch_decode(generated_token_idx_list)

def generate_text_from_tokens(
        model: nn.Module,
        tokens: list[int],
        device: torch.device,
        seq_len: int,
        generation_length: int=100) -> list[str]:
    """ Use model to generate text given beginning input
    Args:
        model (nn.Module): Model used to make predictions.
        tokenizer (Tokenizer): Tokenizer object to handle conversions with
            tokens and strings.
        start_string_list (list[str]): List of strings that should be used as
            beginnings of generated strings (prompts to the model).
        device (torch.device): Device on which to run inference.
        seq_len (int): Context window/sequence length.
        generation_length (int): Total amount of tokens per generated sequence.

    Returns:
        A list of all the fully generated strings by the model.
    """
    # Keep track of fully generated token indices lists
    generated_token_idx_list = []

    # Move model to device
    model = model.to(device)

    # No gradients needed
    model.eval()
    with torch.inference_mode():
        # Retrieve string's tokenized version
        input_token_idxs = tokens

        # Store generated sequence token indices
        generated_token_idxs = input_token_idxs

        # Get tensor with token indices
        input_tensor = torch.tensor(input_token_idxs, dtype=torch.long)

        # Add batch dimension and move to device
        input_tensor = input_tensor.unsqueeze(0).to(device)

        times = []
        # Keep generating until padding or reached generation length
        for _ in range(generation_length):
            # Make sure input_tensor isn't longer than sequence length
            input_tensor = input_tensor[
                :,
                max(0, input_tensor.shape[-1] - seq_len):]

            # Get model predictions
            start = time.time()
            predictions = model(input_tensor)
            end = time.time()
            times.append(end - start)

            # Apply softmax to predictions
            predictions = F.softmax(predictions, dim=-1)

            # Get the last predicted word
            predicted_id = predictions.argmax(dim=-1)[0, -1]

            # Add predicted word to input (used as next input sequence)
            input_tensor = torch.cat(
                [input_tensor, predicted_id[None, None]],
                dim=-1)

            # Store predicted token as part of generation
            generated_token_idxs.append(predicted_id.item())

        print('tokens generated: ', len(times))
        print('time taken: ', sum(times))
        return sum(times)


class Struct():
    def __init__(self, **entries):
        self.config_dict = entries
        for key, value in entries.items():
            setattr(self, key, value)

    def __setattr__(self, name: str, value) -> None:
        if name != "config_dict":
            self.config_dict[name] = value
        super().__setattr__(name, value)

    def __str__(self):
        s = "Struct: {"
        for key, value in self.config_dict.items():
            s += f"{key}: {value},"
        s = s[:-1] + "}"
        return s

    def get_config_dict(self):
        return self.config_dict
