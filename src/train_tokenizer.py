import dask
import dask.dataframe as dd
import sys
import yaml

from pathlib import Path
from tokenizers import decoders, pre_tokenizers, processors, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from utils import Struct

dask.config.set({"dataframe.query-planning": True})

def train_tokenizer(config):
    
    print(f"Data dir: {config.raw_dataset_path}")
    print("Loading dataset from disk")

    train_dataset_path = Path(config.raw_dataset_path) / "train" / "*.parquet"

    # Only load in train set, as that's all the tokenizer needs.
    dataset = dd.read_parquet(path=train_dataset_path,
                              columns=config.dataset_feature) 

    print("Loaded!")

    print("Creating tokenizer")
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

    print("Training tokenizer")
    # Train tokenizer on only training data

    tokenizer.train_from_iterator(
        iter(dataset[config.dataset_feature]), #[:subset_size]
        trainer=trainer)

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

    print("Saving tokenizer to file...")
    # Save tokenizer to file
    tokenizer_save_path = Path(config.tokenizer_path)
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_save_path)
    print("Done!")


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    print("Training tokenizer...")
    train_tokenizer(config)
    