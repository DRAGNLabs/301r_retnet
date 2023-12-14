import torch
import torch.nn.functional as F

from datasets import Tokenizer
from torch import nn


def generate_text(model: nn.Module,
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
    # Keep track of strings after fully generated
    generated_strings = []

    # Move model to device
    model = model.to(device)

    # No gradients needed
    model.eval()
    with torch.inference_mode():
        # Iterate over each start string
        for input_idx in range(len(start_string_list)):
            # Convert initial start string to token indices
            input_token_idxs = tokenizer.stoi(start_string_list[input_idx])

            # Store generated sequence token indices
            generated_token_idxs = input_token_idxs

            # Get tensor with token indices
            input_tensor = torch.tensor(input_token_idxs, dtype=torch.long)

            # Add batch dimension and move to device
            input_tensor = input_tensor.unsqueeze(0).to(device)

            # Generate rest of the string
            while len(generated_token_idxs) < generation_length:
                # Make sure input_tensor isn't longer than sequence length
                input_tensor = input_tensor[:,
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

            # Get text representation of fully generated string and save
            generated_string = " ".join(tokenizer.itos(generated_token_idxs))
            generated_strings.append(generated_string)

    return generated_strings
