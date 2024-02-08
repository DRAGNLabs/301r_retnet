from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from datasets import load_dataset
from utils import Struct

class DataModule(LightningDataModule):
    """
    DataModule class for Lightning. This class is used to load and prepare the
    dataset for training, validation, and testing. It also provides the
    dataloaders for each of the three stages.
    """
    def __init__(self, config: Struct):
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.tokenized_dataset_path = config.tokenized_dataset_path

    def setup(self, stage: str):
        if stage == "fit":
            # Load datasets
            train_tokenized_dataset = load_dataset(
                "parquet",
                data_files=str(
                    Path(self.tokenized_dataset_path) / "train.parquet"),
                split="all")
            val_tokenized_dataset = load_dataset(
                "parquet",
                data_files=str(
                    Path(self.tokenized_dataset_path) / "validation.parquet"),
                split="all")

            # Convert datasets into PyTorch format
            self.train_dataset = \
                train_tokenized_dataset.with_format("torch")["input_ids"]
            self.val_dataset = \
                val_tokenized_dataset.with_format("torch")["input_ids"]

        if stage == "test":
            # Load dataset
            test_tokenized_dataset = load_dataset(
                "parquet",
                data_files=str(
                    Path(self.tokenized_dataset_path) / "test.parquet"),
                split="all")

            # Convert datasets into PyTorch format
            self.test_dataset = \
                test_tokenized_dataset.with_format("torch")["input_ids"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
