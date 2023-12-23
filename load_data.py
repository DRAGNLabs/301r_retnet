from datasets import get_dataset_split_names, load_dataset
from os import environ
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.normalizers import NFD, Sequence, Strip, StripAccents
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

# Disable parallelism to avoid errors with DataLoaders later on
environ["TOKENIZERS_PARALLELISM"] = "false"

def get_loaders_tokenizer(dataset_name: str,
                          tokenizer_name: str,
                          seq_len: int,
                          batch_size: int,
                          vocab_size: int,
                          data_dir: str,
                          dataset_config: str=None,
                          max_token_len: int=20) -> tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    """ Loads the WikiText2 dataset and returns DataLoaders.
    Args:
        seq_len (int): Context window/sequence length.
        batch_size (int): Batch size.
        vocab_size (int): Maximum vocabulary size.

    Returns:
        Tuple with the format: (Training DataLoader, Validation DataLoader,
        Testing DataLoader, Tokenizer object).
    """
    assert ["test", "train", "validation"] == sorted(get_dataset_split_names(path=dataset_name, config_name=dataset_config)), \
        f"The dataset {dataset_name} doesn't have a train, validation, and test split!"

    # Retrieve iterators for each split of the dataset
    entire_dataset = load_dataset(path=dataset_name, name=dataset_config, cache_dir=data_dir)

    # Function to filter out undesired inputs. In this case, filter out instances with only whitespace
    filter_fun = lambda instance_dict : bool(instance_dict["text"].strip())

    # Filter out undesired data instances
    entire_dataset = entire_dataset.filter(filter_fun)

    # Create one of the four tokenizer models supported by Hugging Face
    if tokenizer_name == "BPE":
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        trainer = BpeTrainer(vocab_size=vocab_size,
                             show_progress=True,
                             special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
                             max_token_length=max_token_len)
    elif tokenizer_name == "Unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(vocab_size=vocab_size,
                                 show_progress=True,
                                 special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
                                 unk_token="<unk>",
                                 max_piece_length=max_token_len)
    elif tokenizer_name == "WordLevel":
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        trainer = WordLevelTrainer(vocab_size=vocab_size,
                                   show_progress=True,
                                   special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
    elif tokenizer_name == "WordPiece":
        tokenizer = Tokenizer(WordPiece(unk_token="<unk>",
                                        max_input_chars_per_word=max_token_len))
        trainer = WordPieceTrainer(vocab_size=vocab_size,
                                   show_progress=True,
                                   special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])

    # Create normalizer
    tokenizer.normalizer = Sequence([NFD(), Strip(), StripAccents()])

    # Enable padding
    tokenizer.enable_padding(direction="right",
                             pad_id=0,
                             pad_token="<pad>",
                             length=seq_len + 1)

    # Enable truncation
    tokenizer.enable_truncation(max_length=seq_len + 1, direction="right")

    # Pretokenization?
    # tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train_from_iterator(entire_dataset["train"].to_iterable_dataset(), trainer=trainer, length=len(entire_dataset["train"]))

    # Wrap tokenizer with transformers library
    tokenizer = PreTrainedTokenizerFast(model_max_length=seq_len,
                                        padding_side="right",
                                        truncation_side="right",
                                        bos_token="<bos>",
                                        eos_token="<eos>",
                                        unk_token="<unk>",
                                        pad_token="<pad>",
                                        tokenizer_object=tokenizer)

    # Tokenize the datasets
    tokenization = lambda instances_dict : \
            tokenizer(instances_dict["text"],
                      padding="max_length",
                      truncation=True,
                      max_length=seq_len + 1,
                      return_token_type_ids=False,
                      return_attention_mask=False)

    entire_dataset = entire_dataset.map(tokenization, batched=True)

    # Drop now unnecessary "text" column
    entire_dataset = entire_dataset.remove_columns(column_names="text")

    # Create DataLoaders
    train_loader = DataLoader(entire_dataset["train"].with_format("torch")["input_ids"], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(entire_dataset["validation"].with_format("torch")["input_ids"], batch_size=batch_size)
    test_loader = DataLoader(entire_dataset["test"].with_format("torch")["input_ids"], batch_size=batch_size)

    return train_loader, valid_loader, test_loader, tokenizer
