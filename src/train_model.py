# General
import json
import os
import shutil
import signal
import sys
import torch
import yaml

from codecarbon import OfflineEmissionsTracker
from dataset import DataModule
from datetime import datetime
from models import RetNetModel, TransformerModel
from pathlib import Path
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.plugins.environments import SLURMEnvironment
from subprocess import run
from tabulate import tabulate
from transformers import set_seed
from torchinfo import summary as model_summary
from utils import Struct

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath, monitor, save_top_k, mode, every_n_train_steps):
        self.num_ckpts = 0
        self.file_name = f"{self.num_ckpts}"+"_epoch_{epoch}_validation_{val_loss:.2f}"
        
        super().__init__(
            dirpath=dirpath,
            filename=self.file_name,
            monitor=monitor,
            save_top_k=save_top_k,
            mode=mode,
            every_n_train_steps=every_n_train_steps)


    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(
            trainer=trainer,
            pl_module=pl_module,
            checkpoint=checkpoint)
        pl_module.save_pretrained(
            os.path.join(self.dirpath, f"hf_ckpt_{self.num_ckpts}"))
        self.num_ckpts += 1

        # Print GPU memory usage
        print(torch.cuda.memory_summary())  # Prints per device

def train_model(config: Struct):
    # Test that the head dimension will be an even, whole number
    assert config.embed_dim % (config.heads * 2) == 0, \
        "Head Dimension must be even to perform Rotary Position Embedding " + \
        f"({config.embed_dim} / {config.heads} = " + \
        f"{config.embed_dim / config.heads} -- not an even, whole number)! " + \
        "Try changing the Embedding Dimension or number of heads."
    
    # Test that the value embedding dimension is divisible by number of heads
    assert config.value_embed_dim % config.heads == 0, \
        "Value Embed Dimension not divisible by number of heads " + \
        f"({config.value_embed_dim} % {config.heads} != 0)!"

    # Set random seeds for torch, numpy, random, etc. with transformers library
    if config.rand_seed is not None:
        set_seed(config.rand_seed)

    # Create requested model
    if config.model_type.lower() == "retnet":
        model = RetNetModel(config)
    elif config.model_type.lower() == "transformer":
        model = TransformerModel(config)
    else:
        raise ValueError(f"Model type '{config.model_type}' not supported!")

    # Print all arguments for recordkeeping
    print("Arguments:")
    arg_table = []
    row = []
    for i, (key, value) in enumerate(config.get_config_dict().items()):
        row.append(f"{key}: {value}")
        if (i + 1) % 4 == 0:
            arg_table.append(row)
            row = []
    if row:
        arg_table.append(row)
    print(tabulate(arg_table, tablefmt="grid"))

    # Print model info
    print("\nModel Summary:")
    total_params = model_summary(
        model,
        input_data=torch.ones(1, config.seq_len).long()).total_params

    # Create unique label for model (model type, parameter count,
    # **hyperparameters, timestamp)
    model_label = f"{config.model_type}_{total_params}" + \
        f"_LR{config.learning_rate}_ED{config.embed_dim}" + \
        f"_FFN{config.ffn_dim}_H{config.heads}_S{config.seq_len}" + \
        f"_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"

    # Initialize model directory for config files, weights, etc.
    model_dir = Path(config.models_path) / model_label
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving model files in {model_dir}")

    # Initialize checkpoints directory
    checkpoints_dir = model_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=False, exist_ok=True)
    print(f"Saving checkpoints in {checkpoints_dir}")

    # Create SummaryWriter to record logs for TensorBoard
    tboard_log_dir = Path(config.models_path) / model_label / "logs"
    
    print(f"Saving TensorBoard logs in {tboard_log_dir}")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=tboard_log_dir)

    # Save all the variables in args as JSON inside folder
    json.dump(
        obj=arg_table,
        fp=open(model_dir / "model_args.json", "w"),
        indent=4)

    # Print estimated loss if it hasn't learned anything
    print("\nEstimated Loss if guessing:")
    print(f"-log(1 / {config.vocab_size}) = " + \
        f"{-torch.log(torch.tensor(1 / config.vocab_size))}")

    # Checks to see if you want to use a portion of the dataset for training
    if (config.split_dataset != 1.0):
        print(f"\nUsing {config.split_dataset * 100}% of the dataset for training")

        # Define the source and target directories
        source_dir = Path(config.tokenized_dataset_path + "/train")
        target_dir = Path(config.tokenized_dataset_path + "/portion_data")

        # Ensure the target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)

        # Calculate the number of files to copy based on the percentage
        total_files = len(list(source_dir.glob('*.parquet')))
        files_to_copy = int((config.split_dataset) * total_files)

        # Count the existing files in the target directory
        existing_files_count = len(list(target_dir.glob('*.parquet')))
        print(f"\nThere are {existing_files_count} .parquet files available, of which {files_to_copy} files are being copied which is actually {(files_to_copy / total_files)*100:.4f}% of the training data, instead of {config.split_dataset * 100}%")

        # Check if the existing number of files matches the expected number of files
        if existing_files_count == files_to_copy:
            print(f"The specified portion of the dataset ({config.split_dataset * 100}%) is already present in {target_dir}. No changes made.")
        else:
            print(f"Preparing to copy {files_to_copy} files to {target_dir}.")

            # Clear the target directory if the file counts do not match
            for item in target_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except FileNotFoundError as e:
                    print(f"Warning: Tried to delete {item}, but file was not found.")

            # Proceed to copy the calculated number of .parquet files
            for i in range(files_to_copy):
                source_file = source_dir / f"part.{i}.parquet"
                if source_file.exists():  # Check if the source file exists
                    target_file = target_dir / f"part.{i}.parquet"
                    shutil.copy2(source_file, target_file)
                else:
                    print(f"Warning: File {source_file} does not exist and will not be copied.")
                    
            print(f"Copied {files_to_copy} files to {target_dir}.")

    # Loads Tokenized data
    print(f"\nNow loading '{config.dataset_name}'")

    dm = DataModule(config)

    # Implement callbacks
    model_checkpoint = CustomModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor="val_loss",
        save_top_k=config.save_top_k,
        mode="min",
        every_n_train_steps=config.every_n_train_steps)

    early_stopping = EarlyStopping(
        "val_loss",
        patience=config.early_stopping,
        mode="min",
        verbose=True)

    # Setup Trainer based on if using Slurm or not
    if not config.use_slurm:
        trainer = Trainer(
            default_root_dir=model_dir, # main directory for run
            accelerator=config.device,
            devices=config.num_devices,
            strategy=config.strategy,
            max_epochs=config.epochs,
            val_check_interval=config.val_check_interval,
            accumulate_grad_batches=config.accumulate_grad_batches,
            sync_batchnorm=True,
            callbacks=[early_stopping, model_checkpoint],
            logger=tb_logger,
            precision=config.precision)
    else:
        trainer = Trainer(
            default_root_dir=model_dir, # main directory for run
            accelerator=config.device,
            num_nodes=config.num_nodes,
            devices=config.num_devices,
            strategy=config.strategy,
            max_epochs=config.epochs,
            val_check_interval=config.val_check_interval,
            accumulate_grad_batches=config.accumulate_grad_batches,
            sync_batchnorm=True,
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
            callbacks=[early_stopping, model_checkpoint],
            logger=tb_logger,
            precision=config.precision)
        
    ## Set up carbon emissions tracker

    CO2_outfile = "emissions.txt" if not config.CO2_outfile else config.CO2_outfile
    emissions_tracker = OfflineEmissionsTracker(
                output_dir=model_dir,
                output_file=CO2_outfile,
                country_iso_code="USA",
                cloud_provider="gcp",  # As of March 13, 2024, GCP us-west is the region with the most similar consumption profile to BYU.
                cloud_region="us-west3")

    emissions_tracker.start()
    trainer.fit(model, datamodule=dm)

    print("\nDone training! Now testing model...")

    # Automatically load best checkpoint and test with test dataloader
    trainer.test(model, datamodule=dm)
    emissions_tracker.stop()

    print("Finished training!")

    # Retrieve info of the best checkpoint file
    best_model_path = model_checkpoint.best_model_path
    best_model_score = model_checkpoint.best_model_score
    print(f"Best Checkpoint File Path: {best_model_path}")
    print(f"Best Model Score: {best_model_score}")

    return best_model_path, best_model_score


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    train_model(config)
