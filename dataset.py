from pathlib import Path
from pytorch_lightning import LightningDataModule
from typing import Optional
from torch.utils.data import DataLoader

from datasets import load_dataset

class DataModule(LightningDataModule):
    def __init__(self, config, num_workers=0):
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = num_workers
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
