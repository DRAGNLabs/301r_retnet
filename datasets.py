import torch
import torch.nn.functional as F

from collections.abc import Callable
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator, Vocab


class Tokenizer():
    """ Handles all interactiions with underlying tokenizer and vocab. """
    def __init__(self, vocab: Vocab, tokenizer_fun: Callable[[str], list[str]]):
        """
        Args:
            vocab (Vocab): A Vocab object created from the training data text.
            tokenizer_fun (Callable[[str], list[int]]): A function that takes in
                a string and returns a list of strings, each of them a token.
        """
        self.vocab = vocab
        self.tokenizer_fun = tokenizer_fun

    def stoi(self, text: str) -> list[int]:
        """ Return token indices for tokens in text.
        Args:
            text (str): String to split into tokens and get indices for.

        Returns:
            A list of ints corresponding to the indices of the tokens in text.
        """
        # Convert input string into list of tokens (strings)
        tokens = self.tokenizer_fun(text)

        return self.vocab.lookup_indices(tokens)

    def itos(self, token_idxs: list[int]) -> list[str]:
        """ Return list of tokens corresponding to given indices.
        Args:
            token_idxs (list): List of ints that are the indices of desired
                tokens in vocab.

        Returns:
            A list of tokens corresponding to the token indices given.
        """
        return self.vocab.lookup_tokens(token_idxs)


def get_vocab_tokenizer(tokenizer_fun_name: str,
                        training_iter: IterDataPipe,
                        vocab_size: int) -> tuple[Vocab, Tokenizer]:
    """ Returns instance of Tokenizer based of given parameters.
    Args:
        tokenizer_fun_name (str): Name of tokenizer function.
        training_iter (IterDataPipe): Iterator that returns instances from the
            training data.
        vocab_size (int): Maximum vocabulary size.

    Returns:
        A tuple with a Vocab object created from the training_iter and a
        Tokenizer object built around the given tokenizer function.
    """
    # Tokenizer function
    tokenizer_fun = get_tokenizer(tokenizer_fun_name)

    # Inner function to yield tokens from dataset iterator
    def yield_tokens(data_iterator: IterDataPipe) -> list[str]:
        """ Yields lists of tokens from entries in IterDataPipe.
        Args:
            data_iterator (IterDataPipe): Iterator that returns string instances
                from data.

        Yields:
            A list of token strings of the next element in data_iterator.
        """
        for string in data_iterator:
            yield tokenizer_fun(string)

    # Build vocabulary from training iterator
    vocab = build_vocab_from_iterator(
            yield_tokens(training_iter),
            specials=["<pad>", "<unk>"],
            max_tokens=vocab_size)
    vocab.set_default_index(vocab["<unk>"])

    return vocab, Tokenizer(vocab, tokenizer_fun)


def iter_to_tensors(raw_text_iter: IterDataPipe,
                    seq_len: int,
                    tokenizer: Tokenizer,
                    vocab_padding_idx: int,
                    skip_whitespace_inputs: bool=True) -> tuple[Tensor, Tensor]:
    """ Convert data in iterator into a padded tensor representation.
    Args:
        raw_text_iter (IterDataPipe): Iterator that returns string instances
            from data.
        seq_len (int): Context window/sequence length.
        tokenizer (Tokenizer): Tokenizer object to handle string to token
            indices conversions.
        vocab_padding_idx (int): The vocabulary's index of the padding token.
        skip_whitespace_inputs (bool): If True then won't add string instances
            from the raw_text_iter that are completely whitespace to the tensor.

    Returns:
        A tuple of two Tensor objects, each of dimensions (number of elements in
	raw_text_iter, seq_len), where the first is of inputs and the second is
	of the corresponding targets.
    """
    padded_input_tensors = []
    for text in raw_text_iter:
        # Skip element if text is all whitespace and we want to ignore pure
        # whitespace inputs
        if skip_whitespace_inputs and text.isspace():
            continue

        # Tokenize and get corresponding token indices
        text_token_idxs = tokenizer.stoi(text)

        # NOTE: The following code adds an extra token past the sequence length
        # that will be part of the targets needed during training

        # Truncate tokens if longer than sequence length + extra token needed
        # for target
        text_token_idxs = text_token_idxs[:seq_len + 1]

        # Convert to tensor and add padding
        padded_token_idxs = F.pad(
                input=torch.tensor(text_token_idxs, dtype=torch.long),
                pad=(0, seq_len + 1 - len(text_token_idxs)),
                value=vocab_padding_idx)

        padded_input_tensors.append(padded_token_idxs)

    # Combine all tensor elements into one large tensor of dimensions (number of
    # elements in raw_text_iter, seq_len + 1)
    combined_padded_tensors = torch.stack(padded_input_tensors)

    return combined_padded_tensors[:, :-1], combined_padded_tensors[:, 1:]


def load_wikitext2(seq_len: int,
		   batch_size: int,
		   vocab_size: int) -> tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    """ Loads the WikiText2 dataset and returns DataLoaders.
    Args:
        seq_len (int): Context window/sequence length.
        batch_size (int): Batch size.
        vocab_size (int): Maximum vocabulary size.

    Returns:
        Tuple with the format: (Training DataLoader, Validation DataLoader,
        Testing DataLoader, Tokenizer object).
    """
    # Load the WikiText2 dataset and stores corresponding IterDataPipe objects
    train_iter, valid_iter, test_iter = WikiText2()

    vocab, tokenizer = get_vocab_tokenizer(
            tokenizer_fun_name="basic_english",
            training_iter=train_iter,
            vocab_size=vocab_size)

    # Process the datasets
    train_inputs, train_targets = iter_to_tensors(
            raw_text_iter=train_iter,
            seq_len=seq_len,
            tokenizer=tokenizer,
            vocab_padding_idx=vocab.get_default_index(),
            skip_whitespace_inputs=True)

    valid_inputs, valid_targets = iter_to_tensors(
            raw_text_iter=valid_iter,
            seq_len=seq_len,
            tokenizer=tokenizer,
            vocab_padding_idx=vocab.get_default_index(),
            skip_whitespace_inputs=True)

    test_inputs, test_targets = iter_to_tensors(
            raw_text_iter=test_iter,
            seq_len=seq_len,
            tokenizer=tokenizer,
            vocab_padding_idx=vocab.get_default_index(),
            skip_whitespace_inputs=True)

    # Create Datasets
    train_dataset = TensorDataset(train_inputs, train_targets)
    valid_dataset = TensorDataset(valid_inputs, valid_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, tokenizer
