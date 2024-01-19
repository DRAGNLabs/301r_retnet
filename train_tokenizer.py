from datasets import (
    DatasetDict,
    get_dataset_infos as get_ds_infos,
    get_dataset_split_names as get_ds_split_names,
    load_dataset as load_ds)
from os import environ
from tokenizers import decoders, pre_tokenizers, processors, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from argparse import ArgumentParser
from pathlib import Path

# Disable parallelism to avoid errors with DataLoaders later on
environ["TOKENIZERS_PARALLELISM"] = "false"

def get_loaders_tokenizer(
        tokenizer_folder: str,
        dataset_name: str,
        seq_len: int,
        vocab_size: int,
        dataset_dir: str,
        dataset_subset: str=None,
        text_feature: str="text",
        splits: list[float]=[0.7, 0.2, 0.1],
        rand_seed: int=None) -> \
            tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    """ Loads Hugging Face dataset and creates DataLoaders and Tokenizer.
    Args:
        dataset_name (str): Name of Hugging Face dataset.
        seq_len (int): Context window/sequence length.
        batch_size (int): Batch size.
        vocab_size (int): Maximum vocabulary size.
        dataset_dir (str): Relative path from base of repository to directory in
            which to download the dataset.
        dataset_subset (str): Configuration/subset of dataset to use.
        text_feature (str): Name of the feature/column of the dataset to use.
       
        splits (list[float]): A list of three floats containing the train,
            validation, and test splits respectively. Should sum to 1.
        rand_seed (int): Seed used during dataset shuffling, ignored if None.

    Returns:
        Tuple with the format: (Training DataLoader, Validation DataLoader,
        Testing DataLoader, Tokenizer object).
    """
    # Test text_feature is actually a feature of the dataset
    
    # Retrieve iterators for each split of the dataset
    print(f'Data dir: {dataset_dir}')

    entire_dataset = load_ds(
        "parquet",
        data_files=str(Path(dataset_dir) / dataset_name / f"{dataset_subset}.parquet"),
        split="all")

    # Function to filter out undesired inputs. In this case, filter out
    # instances with only whitespace
    filter_fun = lambda inst_dict : bool(inst_dict[text_feature].strip())

    # Filter out undesired data instances
    entire_dataset = entire_dataset.filter(filter_fun)

    # Split into training, validation, and testing datasets
    train_validtest = entire_dataset.train_test_split(
        train_size=splits[0],
        shuffle=True,
        seed=rand_seed)
    valid_test = train_validtest["test"].train_test_split(
        train_size=splits[1] / (splits[1] + splits[2]),
        shuffle=True,
        seed=rand_seed)
    entire_dataset = DatasetDict({
        "train": train_validtest["train"],
        "validation": valid_test["train"],
        "test": valid_test["test"]})

    # Create BytePair Encoding tokenizer and trainer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["<pad>", "<bos>", "<unk>"])

    # Like GPT-2, we skip the normalizer and go directly to pre-tokenization.
    # The option we add to ByteLevel here is to not add a space at the beginning
    # of a sentence (which is the default otherwise)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Train tokenizer on only training data
    tokenizer.train_from_iterator(
        iter(entire_dataset["train"][text_feature]),
        trainer=trainer,
        length=len(entire_dataset["train"]))

    # trim_offsets=False tells post-processor to keep spaces as part of tokens
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A",
        special_tokens=[("<bos>", tokenizer.token_to_id("<bos>"))],
    )

    # Add decoder for converting tokens back to text
    tokenizer.decoder = decoders.ByteLevel()

    # Enable padding
    tokenizer.enable_padding(
        direction="right",
        pad_id=0,
        pad_token="<pad>",
        length=seq_len + 1)

    # Enable truncation
    tokenizer.enable_truncation(max_length=seq_len + 1, direction="right")

    # Wrap tokenizer with transformers library
    tokenizer = PreTrainedTokenizerFast(
        model_max_length=seq_len,
        padding_side="right",
        truncation_side="right",
        bos_token="<bos>",
        unk_token="<unk>",
        pad_token="<pad>",
        tokenizer_object=tokenizer)

    # Save tokenizer to file
    tokenizer.save_pretrained(tokenizer_folder)

if __name__ == "__main__":
    # Get arguments
    parser = ArgumentParser()

    parser.add_argument(
        "--tokenizer_folder",
        type=str,
        required=True,
        help="Folder to save tokenizer to.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of Hugging Face dataset.")
    parser.add_argument(
        "--seq_len",
        type=int,
        required=True,
        help="Context window/sequence length.")
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Maximum vocabulary size.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Relative path from base of repository to directory in which to download the dataset.")
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default=None,
        help="Configuration/subset of dataset to use.")
    parser.add_argument(
        "--text_feature",
        type=str,
        default="text",
        help="Name of the feature/column of the dataset to use.")
    parser.add_argument("--splits", type=float, nargs=3,
        default=[0.7, 0.2, 0.1],
        help="Space-separated decimal splits of train, validation, and " + \
            "test datasets. (Ex: '0.7 0.2 0.1')")
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=None,
        help="Seed used during dataset shuffling, ignored if None.")

    args = parser.parse_args()

    get_loaders_tokenizer(args.tokenizer_folder, args.dataset_name, args.seq_len, args.vocab_size, args.dataset_dir, args.dataset_subset, args.text_feature, args.splits, args.rand_seed)
