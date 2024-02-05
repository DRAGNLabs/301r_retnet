from pathlib import Path
from pytorch_lightning import LightningDataModule
from typing import List, Optional
from torch.utils.data import DataLoader

from datasets import load_dataset

class DataModule(LightningDataModule):
    def __init__(self, config, num_workers=0):
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = num_workers

        # Load tokenized datasets
        self.train_tokenized_dataset = load_dataset(
            "parquet",
            data_files=str(Path(config.tokenized_dataset_path) / "train.parquet"),
            split="all")
        self.val_tokenized_dataset = load_dataset(
            "parquet",
            data_files=str(Path(config.tokenized_dataset_path) / "validation.parquet"),
            split="all")
        self.test_tokenized_dataset = load_dataset(
            "parquet",
            data_files=str(Path(config.tokenized_dataset_path) / "test.parquet"),
            split="all")
    
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self.train_tokenized_dataset.with_format("torch")["input_ids"]
        self.val_dataset = self.val_tokenized_dataset.with_format("torch")["input_ids"]
        self.test_dataset = self.test_tokenized_dataset.with_format("torch")["input_ids"]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
