import datasets
import sys
import yaml

from datasets import DatasetDict
from pathlib import Path
from utils import Struct

def download_data(config):
    """ Download dataset from Hugging Face.

    It is useful to download the dataset before trying to train the model when
    the training will take place in a location without access to the internet.

    Args:
        dataset_name (str): Name of Hugging Face dataset.
        dataset_subset (str): Configuration/subset of dataset to use.
        datasets_dir (str): Absolute path to the directory in which Hugging Face
            datasets are downloaded.
    """
    datasets.logging.set_verbosity_debug()
    # what is 800 gb in bytes?
    # 800 * 1024 * 1024 * 1024
    # 858993459200
    datasets.config.IN_MEMORY_MAX_SIZE = 858993459200

    # Create folder to save this dataset's files in
    dataset_dir = Path(config.raw_dataset_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("Beginning download")
    print(f"File path: {dataset_dir}")
    print(f"Data name: {config.dataset_name}")
    print(f"Data subset: {config.dataset_subset}")
    dataset = datasets.load_dataset(
            path=config.dataset_name,
            name=config.dataset_subset,
            split='all',
            trust_remote_code=True,
            num_proc=config.num_proc,
            keep_in_memory=True,
            cache_dir=config.cache_dir)
    
    print('FILTERING')
    # Function to filter out undesired inputs. In this case, filter out
    # instances with only whitespace
    filter_fun = lambda inst_dict : bool(
        inst_dict[config.dataset_feature].strip())

    # Filter out undesired data instances
    dataset = dataset.filter(filter_fun, num_proc=config.num_proc)

    print('SPLITTING')
    # Split into training, validation, and testing datasets
    train_validtest = dataset.train_test_split(
        train_size=config.splits[0],
        shuffle=True,
        seed=config.rand_seed)
    valid_test = train_validtest["test"].train_test_split(
        train_size=config.splits[1] / (config.splits[1] + config.splits[2]),
        seed=config.rand_seed)
    dataset = DatasetDict({
        "train": train_validtest["train"],
        "validation": valid_test["train"],
        "test": valid_test["test"]})

    print('SAVING TO DISK')
    # Save splits to file
    dataset.save_to_disk(
        dataset_dict_path=Path(config.raw_dataset_path),
        num_proc=config.num_proc)

    # # Check if dataset is of type datasets.arrow_dataset.Dataset
    # if isinstance(dataset, datasets.arrow_dataset.Dataset):
    #     filename = config.dataset_subset + ".parquet"
    #     dataset.to_parquet(dataset_dir / filename)
    # else:
    #     raise Exception("Dataset is not of type " + \
    #         "datasets.arrow_dataset.Dataset or " + \
    #         "datasets.dataset_dict.DatasetDict")

    # for key, value in dataset.items():
    #     # shard dataset into smaller files
    #     num_shards = 500

    #     for i in range(num_shards):
    #         print(f"Sharding {key} dataset, shard {i} of {num_shards}")
    #         shard = value.shard(
    #             num_shards=num_shards,
    #             index=i,
    #             keep_in_memory=True)
    #         shard.to_csv(dataset_dir / (key + f'_shard_{i}.csv'), num_proc=config.num_proc)
    #     #value.to_csv(dataset_dir / (key+'.csv'), num_proc=config.num_proc) #TODO: test this

    print("Download completed.")


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    download_data(config)
