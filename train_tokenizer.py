import sys
import yaml

from datasets import DatasetDict, load_dataset as load_ds
from pathlib import Path
from tokenizers import decoders, pre_tokenizers, processors, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from utils import Struct

def train_tokenizer(config) -> \
            tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    # Retrieve iterators for each split of the dataset
    print(f"Data dir: {config.raw_dataset_path}")
    data_path = Path(config.raw_dataset_path) / \
        (config.dataset_subset + ".parquet")

    entire_dataset = load_ds("parquet",
                             data_files=str(data_path),
                             split="all")

    # Function to filter out undesired inputs. In this case, filter out
    # instances with only whitespace
    filter_fun = lambda inst_dict : bool(
        inst_dict[config.dataset_feature].strip())

    # Filter out undesired data instances
    entire_dataset = entire_dataset.filter(filter_fun)

    # Split into training, validation, and testing datasets
    train_validtest = entire_dataset.train_test_split(
        train_size=config.splits[0],
        shuffle=True,
        seed=config.rand_seed)
    valid_test = train_validtest["test"].train_test_split(
        train_size=config.splits[1] / (config.splits[1] + config.splits[2]),
        seed=config.rand_seed)
    entire_dataset = DatasetDict({
        "train": train_validtest["train"],
        "validation": valid_test["train"],
        "test": valid_test["test"]})

    # Save splits to file
    entire_dataset.save_to_disk(
        dataset_dict_path=Path(config.raw_dataset_path))

    # Create BytePair Encoding tokenizer and trainer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        show_progress=True,
        special_tokens=["<pad>", "<bos>", "<unk>"])

    # Like GPT-2, we skip the normalizer and go directly to pre-tokenization.
    # The option we add to ByteLevel here is to not add a space at the beginning
    # of a sentence (which is the default otherwise)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Train tokenizer on only training data
    tokenizer.train_from_iterator(
        iter(entire_dataset["train"][config.dataset_feature]),
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
        length=config.seq_len + 1)

    # Enable truncation
    tokenizer.enable_truncation(
        max_length=config.seq_len + 1,
        direction="right")

    # Wrap tokenizer with transformers library
    tokenizer = PreTrainedTokenizerFast(
        model_max_length=config.seq_len,
        padding_side="right",
        truncation_side="right",
        bos_token="<bos>",
        unk_token="<unk>",
        pad_token="<pad>",
        tokenizer_object=tokenizer)

    # Save tokenizer to file
    tokenizer_save_path = Path(config.tokenizer_path)
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_save_path)


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    train_tokenizer(config)
