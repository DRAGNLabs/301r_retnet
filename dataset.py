from typing import List, Optional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from datasets import (
    DatasetDict,
    load_dataset as load_ds)

class DataModule(LightningDataModule):
    def __init__(self, train_path, val_path, tokenizer, batch_size, sequence_length, num_workers=0):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        self.entire_dataset = None # TODO: Load datasetdict object from file
    
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self.entire_dataset["train"].with_format("torch")["input_ids"]
        self.val_dataset = self.entire_dataset["validation"].with_format("torch")["input_ids"]
        self.test_dataset = self.entire_dataset["test"].with_format("torch")["input_ids"]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


