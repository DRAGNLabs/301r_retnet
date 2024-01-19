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

# Disable parallelism to avoid errors with DataLoaders later on
environ["TOKENIZERS_PARALLELISM"] = "false"

def get_loaders_tokenizer(
        dataset_name: str,
        seq_len: int,
        batch_size: int,
        vocab_size: int,
        data_dir: str,
        dataset_config: str=None,
        text_feature: str="text",
        max_token_len: int=20,
        splits: list[float]=[0.7, 0.2, 0.1],
        rand_seed: int=None) -> \
            tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    """ Loads Hugging Face dataset and creates DataLoaders and Tokenizer.
    Args:
        dataset_name (str): Name of Hugging Face dataset.
        seq_len (int): Context window/sequence length.
        batch_size (int): Batch size.
        vocab_size (int): Maximum vocabulary size.
        data_dir (str): Relative path from base of repository to directory in
            which to download the dataset.
        dataset_config (str): Configuration/subset of dataset to use.
        text_feature (str): Name of the feature/column of the dataset to use.
        max_token_len (int): Prevents tokenizer creating tokens longer than the
            specified size.
        splits (list[float]): A list of three floats containing the train,
            validation, and test splits respectively. Should sum to 1.
        rand_seed (int): Seed used during dataset shuffling, ignored if None.

    Returns:
        Tuple with the format: (Training DataLoader, Validation DataLoader,
        Testing DataLoader, Tokenizer object).
    """
    # Test text_feature is actually a feature of the dataset

    # Note from Jay: I can't find any docs on this to make it work offline, seems unnecessary if you've already taken the time and care to download the data
    """ds_features = get_ds_infos(
        dataset_name,
        trust_remote_code=True)[dataset_config].features
    assert text_feature in ds_features, \
        f"'{text_feature}' not in '{dataset_name}' features {ds_features}!"
    """
    
    # Retrieve iterators for each split of the dataset
    print(f'Data dir: {data_dir}')
    
    entire_dataset = load_ds("parquet", data_files=str(data_dir) + ".parquet", split="all")

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

    #! # Create BytePair Encoding tokenizer and trainer
    # tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    # trainer = BpeTrainer(
    #     vocab_size=vocab_size,
    #     show_progress=True,
    #     special_tokens=["<pad>", "<bos>", "<unk>"],
    #     max_token_length=max_token_len)

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

    # Tokenize the datasets
    tokenization = lambda instances_dict : \
        tokenizer(
            instances_dict[text_feature],
            padding="max_length",
            truncation=True,
            max_length=seq_len + 1,
            return_token_type_ids=False,
            return_attention_mask=False)

    entire_dataset = entire_dataset.map(tokenization, batched=True)

    # Drop now unnecessary text_feature column
    entire_dataset = entire_dataset.remove_columns(column_names=text_feature)

    # Create DataLoaders
    train_loader = DataLoader(
        entire_dataset["train"].with_format("torch")["input_ids"],
        batch_size=batch_size,
        shuffle=True)
    valid_loader = DataLoader(
        entire_dataset["validation"].with_format("torch")["input_ids"],
        batch_size=batch_size)
    test_loader = DataLoader(
        entire_dataset["test"].with_format("torch")["input_ids"],
        batch_size=batch_size)

    return train_loader, valid_loader, test_loader, tokenizer
