from typing import List, Optional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from datasets import (
    DatasetDict,
    load_dataset as load_ds)
from pathlib import Path

class DataModule(LightningDataModule):
    def __init__(self, train_dataset_path, val_dataset_path, test_dataset_path, batch_size, num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_tokenized_dataset = load_ds("parquet",
                            data_files=train_dataset_path,
                            split="all")
        self.val_tokenized_dataset = load_ds("parquet",
                            data_files=val_dataset_path,
                            split="all")
        self.test_tokenized_dataset = load_ds("parquet",
                            data_files=test_dataset_path,
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


