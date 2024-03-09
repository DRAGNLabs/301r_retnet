import dask
import dask.dataframe as dd
import torch
import psutil
import sys

from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, get_worker_info
from torch.distributed import get_rank, get_world_size
from utils import Struct

dask.config.set({'dataframe.query-planning': True})
sys.path.append("..")

class DataModule(LightningDataModule):
    """
    Custom DataModule class for Lightning. This class is used to load and
    prepare the tokenized dataset for training, validation, and testing. It also
    provides the PyTorch DataLoaders for each of the three stages.
    """
    def __init__(self, config: Struct=None):
        """
        Args:
            config (Struct): A Struct object with all configuration fields.
        """
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.tokenized_dataset_path = Path(config.tokenized_dataset_path)
        self.seq_len = config.seq_len

    def setup(self, stage: str):
        """ Setup for each stage -- called on every process on DDP.
        Args:
            stage (str): Either "fit", "validate", "test", or "predict".
        """
        print('memory usage 1: ', psutil.virtual_memory().percent)
        if stage == "fit":
            # Load datasets
            self.train_dataset = DataSet(self.tokenized_dataset_path / 'train',
                                         self.seq_len)
            
            self.val_dataset = DataSet(self.tokenized_dataset_path / 'validation',
                                        self.seq_len)
            
        if stage == "test":
            # Load dataset
            self.test_dataset = DataSet(self.tokenized_dataset_path / 'test',
                                         self.seq_len)
            
        print('memory usage 2: ', psutil.virtual_memory().percent)
            

    def train_dataloader(self):
        """ Return training PyTorch DataLoader. """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def val_dataloader(self):
        """ Return validation PyTorch DataLoader. """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def test_dataloader(self):
        """ Return testing PyTorch DataLoader. """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)
    
class DataSet(torch.utils.data.IterableDataset):
    def __init__(self, path_to_data, seq_len):
        # Read data with Dask
        self.data = dd.read_parquet(path_to_data / "*.parquet")

        # Get length of df
        self.length = len(self.data)

        self.data = self.data.iterrows()
        self.seq_len = seq_len

    def __len__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        world_size = get_world_size()
        total_processes = num_workers * world_size
        return (self.length // total_processes)

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        world_size = get_world_size()
        process_rank = get_rank()

        for index, item in enumerate(self.data):
            if index % (num_workers * world_size) == (process_rank * num_workers + worker_id):
                item = item[1].values[0].copy()
                x = item[:self.seq_len-1]
                y_true = item[1:self.seq_len]
                yield(x,y_true)