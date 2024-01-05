from datasets import (
    get_dataset_infos as get_ds_infos,
    get_dataset_split_names as get_ds_split_names,
    load_dataset as load_ds)
from os import environ
from tokenizers import decoders, pre_tokenizers, processors, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

# Disable parallelism to avoid errors with DataLoaders later on
environ["TOKENIZERS_PARALLELISM"] = "false"

def get_loaders_tokenizer(dataset_name: str,
                          seq_len: int,
                          batch_size: int,
                          vocab_size: int,
                          data_dir: str,
                          dataset_config: str=None,
                          text_feature: str="text",
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
    dataset_split_names = get_ds_split_names(
            path=dataset_name,
            config_name=dataset_config,
            trust_remote_code=True)
    assert "train" in dataset_split_names \
            and "validation" in dataset_split_names \
            and "test" in dataset_split_names, \
        f"The dataset {dataset_name} doesn't have a train, validation, and test split!"

    ds_features = get_ds_infos(
            dataset_name,
            trust_remote_code=True)[dataset_config].features
    assert text_feature in ds_features, \
        f"'{text_feature}' not in '{dataset_name}' features {ds_features}!"

    # Retrieve iterators for each split of the dataset
    entire_dataset = load_ds(
            path=dataset_name,
            name=dataset_config,
            cache_dir=data_dir,
            trust_remote_code=True)

    # Function to filter out undesired inputs. In this case, filter out instances with only whitespace
    filter_fun = lambda instance_dict : bool(instance_dict[text_feature].strip())

    # Filter out undesired data instances
    entire_dataset = entire_dataset.filter(filter_fun)

    # Create BytePair Encoding tokenizer and trainer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(vocab_size=vocab_size,
                         show_progress=True,
                         special_tokens=["<pad>", "<bos>", "<unk>"],
                         max_token_length=max_token_len)

    # Like GPT-2, we skip the normalizer and go directly to pre-tokenization.
    # The option we add to ByteLevel here is to not add a space at the beginning
    # of a sentence (which is the default otherwise)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    tokenizer.train_from_iterator(iter(entire_dataset["train"][text_feature]),
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
    tokenizer.enable_padding(direction="right",
                             pad_id=0,
                             pad_token="<pad>",
                             length=seq_len + 1)

    # Enable truncation
    tokenizer.enable_truncation(max_length=seq_len + 1, direction="right")

    # Wrap tokenizer with transformers library
    tokenizer = PreTrainedTokenizerFast(model_max_length=seq_len,
                                        padding_side="right",
                                        truncation_side="right",
                                        bos_token="<bos>",
                                        unk_token="<unk>",
                                        pad_token="<pad>",
                                        tokenizer_object=tokenizer)

    # Tokenize the datasets
    tokenization = lambda instances_dict : \
            tokenizer(instances_dict[text_feature],
                      padding="max_length",
                      truncation=True,
                      max_length=seq_len + 1,
                      return_token_type_ids=False,
                      return_attention_mask=False)

    entire_dataset = entire_dataset.map(tokenization, batched=True)

    # Drop now unnecessary text_feature column
    entire_dataset = entire_dataset.remove_columns(column_names=text_feature)

    # Create DataLoaders
    train_loader = DataLoader(entire_dataset["train"].with_format("torch")["input_ids"], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(entire_dataset["validation"].with_format("torch")["input_ids"], batch_size=batch_size)
    test_loader = DataLoader(entire_dataset["test"].with_format("torch")["input_ids"], batch_size=batch_size)

    return train_loader, valid_loader, test_loader, tokenizer
