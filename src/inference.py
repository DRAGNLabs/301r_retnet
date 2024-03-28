import sys
import signal
import torch
import yaml

from dataset import DataModule
from datetime import datetime
from pathlib import Path
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.plugins.environments import SLURMEnvironment
from torchinfo import summary as model_summary
from architecture.lightning.performer import PerformerLightning
from architecture.lightning.retnet import RetNetLightning
from architecture.lightning.transformer import TransformerLightning
from utils import Struct


def inference(config: Struct):
    """
    The purpose of this script is to provide the ability to perform inference on
    a trained model. This script will load a trained model from a checkpoint and
    perform inference on the test set using PyTorch Lightning.

    Args:
        config (Struct): A Struct object with all configuration fields.
    """
    if config.model_type.lower() == "retnet":
        model = RetNetLightning(config)
    elif config.model_type.lower() == "transformer":
        model = TransformerLightning(config)
    elif config.model_type.lower() == "performer":
        model = PerformerLightning(config)
    else:
        raise ValueError(f"Model type '{config.model_type}' not supported!")

    # Create SummaryWriter to record logs for TensorBoard
    if config.tboard_path is None:
        tboard_log_dir = Path(config.model_path_dir) / "logs"
    else:
        tboard_log_dir = f"{config.tboard_path}/" + \
            f"{Path(config.model_path_dir).name}"

    print(f"Saving TensorBoard logs in {tboard_log_dir}")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=tboard_log_dir)

    checkpoint = torch.load(config.checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    dm = DataModule(config)

    if not config.use_slurm:
        trainer = Trainer(
            accelerator=config.device,
            devices=config.num_devices,
            max_epochs=config.epochs,
            val_check_interval=config.val_check_interval,
            accumulate_grad_batches=config.accumulate_grad_batches,
            sync_batchnorm=True,
            logger=tb_logger)
    else:
        trainer = Trainer(
            accelerator=config.device,
            num_nodes=config.num_nodes,
            devices=config.num_devices,
            strategy=config.strategy,
            max_epochs=config.epochs,
            val_check_interval=config.val_check_interval,
            accumulate_grad_batches=config.accumulate_grad_batches,
            sync_batchnorm=True,
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
            logger=tb_logger)

    print("\nTesting model...")
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    inference(config)
