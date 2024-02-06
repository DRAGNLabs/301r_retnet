from datetime import datetime
import sys
import signal
import torch
from torchinfo import summary as model_summary
import yaml

from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning import loggers as pl_loggers

from pathlib import Path
from dataset import DataModule
from models import RetNetModel, TransformerModel
from utils import Struct

def inference(config: Struct):
    """
    The purpose of this script is to essentially provide the ability to
    perform inference on a trained model. This script will load a trained
    model from a checkpoint and perform inference on the test set using PyTorch Lightning.
    """

    if config.model_type.lower() == "retnet":
        model = RetNetModel(config)
    elif config.model_type.lower() == "transformer":
        model = TransformerModel(config)
    else:
        raise ValueError(f"Model type '{config.model_type}' not supported!")
    
    total_params = model_summary(
        model,
        input_data=torch.ones(1, config.seq_len).long()).total_params
    
    # Create unique label for model (model type, parameter count, hyperparameters**)
    model_label = f"inference_{config.model_type}_{total_params}_LR{config.learning_rate}_ED{config.embed_dim}_FFN{config.ffn_dim}_H{config.heads}_S{config.seq_len}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"

    if config.tboard_path is None:
        print("TensorBoard path not specified, saving logs in models directory.")
        tboard_log_dir = Path(config.models_path) / "logs" / model_label
    else:
        tboard_log_dir = f"{config.tboard_path}/{model_label}"

    print(f"Saving TensorBoard logs in {tboard_log_dir}")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=tboard_log_dir)
    
    checkpoint = torch.load(config.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    dm = DataModule(config, num_workers=1)

    if not config.use_slurm:
        trainer = Trainer(
            accelerator=config.device,
            devices=config.num_devices,
            max_epochs=config.epochs,
            val_check_interval=config.val_check_interval,
            accumulate_grad_batches=config.accumulate_grad_batches,
            sync_batchnorm=True,
            logger=tb_logger
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
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
            logger=tb_logger
            )

    print("\nTesting model...")
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    inference(config)