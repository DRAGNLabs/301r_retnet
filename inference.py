from dataset import DataModule

from pytorch_lightning import LightningModule, Trainer
from transformers import PreTrainedTokenizerFast

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import SLURMEnvironment

from models import RetNetModel, TransformerModel

import torch
import yaml
import sys
import os
import signal

from utils import Struct

#TODO: this still doesn't work, needs to be fixed. But not completely necessary, we are running test set at end of training
def inference(config: Struct):
    # if config.model_type.lower() == "retnet":
    #     model = RetNetModel(config)
    # elif config.model_type.lower() == "transformer":
    #     model = TransformerModel(config)
    # else:
    #     raise ValueError(f"Model type '{config.model_type}' not supported!")
    model = RetNetModel.load_from_checkpoint(
            checkpoint_path=config.checkpoint_path,
            #hparams_file="/path/to/experiment/version/hparams.yaml",
            map_location=None,
            )
    
    dm = DataModule(config, num_workers=1)

    if not config.use_slurm:
        trainer = Trainer(
            accelerator=config.device,
            devices=config.num_devices,
            max_epochs=config.epochs,
            val_check_interval=config.val_check_interval,
            accumulate_grad_batches=config.accumulate_grad_batches,
            sync_batchnorm=True
            )
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
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)]
            )

    print("\nDone training! Now testing model...")
    trainer.test(model, datamodule=dm) # Automatically loads best checkpoint, and tests with test dataloader

if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    inference(config)