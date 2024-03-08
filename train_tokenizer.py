import csv
import sys
import yaml

import datasets
from datasets import DatasetDict, load_from_disk, load_dataset
from pathlib import Path
from tokenizers import decoders, pre_tokenizers, processors, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from utils import Struct

import dask
dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd

def train_tokenizer(config):
    
    # Retrieve iterators for each split of the dataset
    print(f"Data dir: {config.raw_dataset_path}")

    print('Loading dataset from disk')
    # data_files = {'train': '/home/jo288/compute/retnet/data/datasets/c4test/test.csv'}
    # entire_dataset = load_dataset('csv',
    #                          data_files=data_files,
    #                          keep_in_memory=True,
    #                          split='all',
    #                          num_proc=config.num_proc,
    #                          cache_dir=config.cache_dir)    
    
    # entire_dataset = load_from_disk(Path(config.raw_dataset_path),
    #                                 keep_in_memory=True)

    dataset = dd.read_parquet(path=Path(config.raw_dataset_path)/ 'train' / '*.parquet') # TODO: consider loading only the text column
                        #   on_bad_lines='skip',
                        #   quoting=csv.QUOTE_NONE,
                        #   sep='\n',
                        #   header=None)

    print('loaded!')

    # test_iterator = iter(dataset[config.dataset_feature])
    # for x in test_iterator:
    #     print(x)
    #     print(type(x))
    #print(next(test_iterator))
                         

    # entire_dataset = datasets.load_dataset(
    #         path=config.raw_dataset_path,
    #         split='train',
    #         trust_remote_code=True,
    #         num_proc=config.num_proc,
    #         #keep_in_memory=True,
    #         cache_dir=config.cache_dir)

    print('Creating tokenizer')
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

    # def batch_iterator(batch_size=1000):
    #     batch = []
    #     for example in entire_dataset:
    #         batch.append(example[config.dataset_feature])
    #         if len(batch) == batch_size:
    #             yield batch
    #             batch = []
    #     if batch:  # yield last batch
    #         yield batch

    print('Training tokenizer')
    # Train tokenizer on only training data
    # subset_size=1
    tokenizer.train_from_iterator(
        iter(dataset[config.dataset_feature]), #[:subset_size]
        #batch_iterator(),
        trainer=trainer) # length=subset_size

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

    print('Saving tokenizer to file...')
    # Save tokenizer to file
    tokenizer_save_path = Path(config.tokenizer_path)
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_save_path)
    print('Done!')

if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    print('Training tokenizer...')
    train_tokenizer(config)
    