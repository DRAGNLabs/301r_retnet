# General
import json
import os
import signal
import sys
import torch
import yaml

from dataset import DataModule
from datetime import datetime
from models import RetNetModel, TransformerModel, LongNetModel
from pathlib import Path
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.plugins.environments import SLURMEnvironment
from tabulate import tabulate
from transformers import set_seed
from torchinfo import summary as model_summary
from utils import Struct

# Allow torch to run float32 matrix multiplications in lower precision for
# better performance while training if hardware is capable
torch.backends.cuda.matmul.allow_tf32 = True

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath, filename, monitor, save_top_k, mode):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            save_top_k=save_top_k,
            mode=mode)
        self.num_ckpts = 0

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(
            trainer=trainer,
            pl_module=pl_module,
            checkpoint=checkpoint)
        pl_module.save_pretrained(
            os.path.join(self.dirpath, f"hf_ckpt_{self.num_ckpts}"))
        self.num_ckpts += 1


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
    elif config.model_type.lower() == "longnet":
        model = LongNetModel(config)
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
        model.to(config.device),
        input_data=torch.ones(1, config.seq_len).long().to(config.device)).total_params

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
    if config.tboard_path is None:
        tboard_log_dir = Path(config.models_path) / "logs" / model_label
    else:
        tboard_log_dir = f"{config.tboard_path}/{model_label}"

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

    # Loads Tokenized data
    print(f"\nNow loading '{config.dataset_name}'")

    dm = DataModule(config)

    # Implement callbacks
    model_checkpoint = CustomModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="epoch_{epoch}_validation_{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=config.save_top_k,
        mode="min")

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
            max_epochs=config.epochs,
            val_check_interval=config.val_check_interval,
            accumulate_grad_batches=config.accumulate_grad_batches,
            sync_batchnorm=True,
            callbacks=[early_stopping, model_checkpoint],
            logger=tb_logger)
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
            logger=tb_logger)

    trainer.fit(model, datamodule=dm)

    print("\nDone training! Now testing model...")

    # Automatically load best checkpoint and test with test dataloader
    trainer.test(model, datamodule=dm)

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
