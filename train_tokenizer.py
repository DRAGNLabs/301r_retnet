from argparse import ArgumentParser
from datasets import (
    DatasetDict,
    load_dataset as load_ds)
from os import environ
from pathlib import Path
from tokenizers import decoders, pre_tokenizers, processors, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

def train_tokenizer(
        dataset_name: str,
        datasets_dir: str,
        seq_len: int,
        tokenizer_folder: str,
        vocab_size: int,
        dataset_subset: str=None,
        rand_seed: int=None,
        splits: list[float]=[0.7, 0.2, 0.1],
        text_feature: str="text") -> \
            tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:

    # Retrieve iterators for each split of the dataset
    print(f"Datasets dir: {datasets_dir}")
    data_path = Path(datasets_dir) / dataset_name / (dataset_subset + ".parquet")
    
    entire_dataset = load_ds("parquet", 
                             data_files=str(data_path),
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
        seed=rand_seed)
    entire_dataset = DatasetDict({
        "train": train_validtest["train"],
        "validation": valid_test["train"],
        "test": valid_test["test"]})

    # Save splits to file
    entire_dataset.save_to_disk(
        dataset_dict_path=Path(datasets_dir) / dataset_name)

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
    tokenizer_save_path = Path(tokenizer_folder)
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_save_path)


if __name__ == "__main__":
    # Get arguments
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of Hugging Face dataset.")
    parser.add_argument(
        "--dataset-subset",
        type=str,
        default=None,
        help="Configuration/subset of dataset to use.")
    parser.add_argument(
        "--datasets-dir",
        type=str,
        required=True,
        help="Relative path from base of repository to directory in which to download the dataset.")
    parser.add_argument(
        "--rand-seed",
        type=int,
        default=None,
        help="Seed used during dataset shuffling, ignored if None.")
    parser.add_argument(
        "--seq-len",
        type=int,
        required=True,
        help="Context window/sequence length.")
    parser.add_argument(
        "--splits", 
        type=float, 
        nargs=3,
        default=[0.7, 0.2, 0.1],
        help="Space-separated decimal splits of train, validation, and " + \
            "test datasets. (Ex: '0.7 0.2 0.1')")
    parser.add_argument(
        "--text-feature",
        type=str,
        default="text",
        help="Name of the feature/column of the dataset to use.")
    parser.add_argument(
        "--tokenizer-folder",
        type=str,
        required=True,
        help="Folder to save tokenizer to.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Maximum vocabulary size.")

    args = parser.parse_args()
    train_tokenizer(**vars(args))
