# General
from datetime import datetime
import json
import os
import signal
import sys
import time
import yaml

# Torch
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary as model_summary

# PyTorch Lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import ModelCheckpoint

# Hugging Face
from datasets import load_dataset
from transformers import set_seed, AutoConfig, AutoModel, AutoModelForCausalLM, PreTrainedTokenizerFast

# Other
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm

# Local
from hugging_face_model import RetNetModelHF, TransformerModelHF
from dataset import DataModule
from torchscale.architecture.config import RetNetConfig, DecoderConfig
from utils import generate_text, Struct

# Allow torch to run float32 matrix multiplications in lower precision for
# better performance while training if hardware is capable
torch.backends.cuda.matmul.allow_tf32 = True

class CustomCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath, filename, save_top_k, monitor, mode):
        super().__init__(dirpath=dirpath, filename=filename, save_top_k=save_top_k, monitor=monitor, mode=mode)
        self.num_ckpts = 0
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer=trainer, pl_module=pl_module, checkpoint=checkpoint)
        pl_module.save_pretrained(os.path.join(self.dirpath, f"hf_ckpt_{self.num_ckpts}"))
        self.num_ckpts += 1


class RetNetModel(LightningModule):
    """ Create model with RetNet architecture. """
    def __init__(self, config: Struct):
        super().__init__()
        self.learning_rate = config.learning_rate

        # Create RetNet configuration
        hf_config = RetNetConfig(
            decoder_embed_dim=config.embed_dim,
            decoder_value_embed_dim=config.value_embed_dim,
            decoder_retention_heads=config.retention_heads,
            decoder_ffn_embed_dim=config.ffn_dim,
            decoder_layers=config.layers,
            dropout=config.dropout,
            activation_dropout=config.activation_dropout,
            vocab_size=config.vocab_size,
            max_seq_len=config.seq_len,
            lr=config.learning_rate)

        self.model_hf = RetNetModelHF(hf_config)

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Long tensor of dimensions: (batch size, sequence
                length).

        Returns:
            A tensor of dimensions: (batch size, sequence length, vocabulary
                size).
        """
        preds = self.model_hf(x)
        return preds
    
    def training_step(self, batch, batch_idx):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        preds = self.model_hf(inputs)

        # Reshape the model predictions for Cross Entropy
        preds = preds.transpose(-2, -1)

        # Calculate loss
        loss = self.loss_fn(preds, targets)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Put validation inputs and targets on device
        val_inputs = batch[:, :-1]
        val_targets = batch[:, 1:]

        # Get validation predictions
        val_predictions = self.model_hf(val_inputs)

        # Reshape the model predictions for Cross Entropy
        val_predictions = val_predictions.transpose(-2, -1)

        # Calculate validation loss
        val_loss = self.loss_fn(val_predictions, val_targets)

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=True)

        return val_loss
    
    def test_step(self, batch, batch_idx):
        # Put validation inputs and targets on device
        test_inputs = batch[:, :-1]
        test_targets = batch[:, 1:]

        # Get validation predictions
        test_predictions = self.model_hf(test_inputs)

        # Reshape the model predictions for Cross Entropy
        test_predictions = test_predictions.transpose(-2, -1)

        # Calculate validation loss
        test_loss = self.loss_fn(test_predictions, test_targets)

        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=True)
        
        return test_loss

    def get_params(self) -> dict:
        """ Get model parameters dictionary. """
        return self.model_hf.get_params()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) # model_hf.decoder_stack
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma) # TODO: Implement this
        return [optimizer]#, [lr_scheduler]
    
    def save_pretrained(self, save_folder: str):
        """ Save model weights and parameters to folder. """
        self.model_hf.save_pretrained(save_folder)

class TransformerModel(LightningModule):
    def __init__(self, config: Struct):
        super().__init__()
        self.learning_rate = config.learning_rate

        # Create Transformer Decoder configuration
        config = DecoderConfig(
            decoder_embed_dim=config.embed_dim,
            decoder_value_embed_dim=config.value_embed_dim,
            decoder_attention_heads=config.attention_heads,
            decoder_ffn_embed_dim=config.ffn_dim,
            decoder_layers=config.layers,
            dropout=config.dropout,
            activation_dropout=config.activation_dropout,
            vocab_size=config.vocab_size)

        self.model_hf = TransformerModelHF(config)

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")


    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Long tensor of dimensions: (batch size, sequence
                length).

        Returns:
            A tensor of dimensions: (batch size, sequence length, vocabulary
                size).
        """
        preds = self.model_hf(x)
        return preds
    
    def training_step(self, batch, batch_idx):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        preds = self.model_hf(inputs)

        # Reshape the model predictions for Cross Entropy
        preds = preds.transpose(-2, -1)

        # Calculate loss
        loss = self.loss_fn(preds, targets)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Put validation inputs and targets on device
        val_inputs = batch[:, :-1]
        val_targets = batch[:, 1:]

        # Get validation predictions
        val_predictions = self.model_hf(val_inputs)

        # Reshape the model predictions for Cross Entropy
        val_predictions = val_predictions.transpose(-2, -1)

        # Calculate validation loss
        val_loss = self.loss_fn(val_predictions, val_targets)

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=True)

        return val_loss
    
    def test_step(self, batch, batch_idx):
        # Put validation inputs and targets on device
        test_inputs = batch[:, :-1]
        test_targets = batch[:, 1:]

        # Get validation predictions
        test_predictions = self.model_hf(test_inputs)

        # Reshape the model predictions for Cross Entropy
        test_predictions = test_predictions.transpose(-2, -1)

        # Calculate validation loss
        test_loss = self.loss_fn(test_predictions, test_targets)

        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=True)
        
        return test_loss

    def get_params(self) -> dict:
        """ Get model parameters dictionary. """
        return self.model_hf.get_params()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model_hf.decoder_stack.parameters(), lr=self.learning_rate)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma)
        return [optimizer]#, [lr_scheduler]
    
    def save_pretrained(self, save_folder: str):
        """ Save model weights and parameters to folder. """
        self.model_hf.save_pretrained(save_folder)

def train_model(config: Struct):
    arg_dict = locals()

    # Test that the head dimension will be an even, whole number
    assert config.embed_dim % (config.heads * 2) == 0, \
        "Head Dimension must be even to perform Rotary Position Embedding " + \
        f"({config.embed_dim} / {config.heads} = {config.embed_dim / config.heads} " + \
        "-- not an even, whole number)! Try changing the Embedding " + \
        "Dimension or number of heads."

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
    for i, arg in enumerate(arg_dict.keys()):
        row.append(f"{arg}: {arg_dict[arg]}")
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
    
    # Create unique label for model (model type, parameter count, hyperparameters**)
    model_label = f"{config.model_type}_{total_params}_LR{config.learning_rate}_ED{config.embed_dim}_FFN{config.ffn_dim}_H{config.heads}_S{config.seq_len}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"

    # Initialize model directory for config files, weights, etc.
    model_dir = Path(config.models_path) / model_label
    model_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving model files in {model_dir}")

    # Initialize checkpoints directory
    checkpoints_dir = model_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=False, exist_ok=False)
    print(f"Saving checkpoints in {checkpoints_dir}")

    # Create SummaryWriter to record logs for TensorBoard
    if config.tboard_path is None:
        tboard_log_dir = Path(config.models_path) / "logs" / model_label
    else:
        tboard_log_dir = f"{config.tboard_path}/logs/{model_label}"

    print(f"Saving TensorBoard logs in {tboard_log_dir}")

    # Save all the variables in args as JSON inside folder
    json.dump(
        obj=arg_table,
        fp=open(model_dir / "model_args.json", "w"),
        indent=4)

    # Print estimated loss if it hasn't learned anything
    print("\nEstimated Loss if guessing:")
    print(f"-log(1 / {config.vocab_size}) = " + \
        f"{-torch.log(torch.tensor(1 / config.vocab_size))}")

    # Get Tokenizer from local directory
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_path)

    # Loads Tokenized data
    print(f"\nNow loading '{config.dataset_name}' and tokenizer...")

    dm = DataModule(config, num_workers=1)
    
    # Implement callbacks
    model_checkpoint = CustomCheckpoint(
        dirpath=checkpoints_dir,
        filename='epoch_{epoch}_validation_{val_loss:.2f}',
        save_top_k=config.save_top_k,
        monitor='val_loss',
        mode='min')
    
    if not config.use_slurm:
        trainer = Trainer(
            default_root_dir=model_dir, # main directory for run
            accelerator=config.device,
            devices=config.num_devices,
            max_epochs=config.epochs,
            val_check_interval=config.val_check_interval,
            accumulate_grad_batches=config.accumulate_grad_batches,
            sync_batchnorm=True,
            callbacks=[model_checkpoint]
            )
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
            callbacks=[model_checkpoint]
            )
    
    trainer.fit(model, datamodule=dm)

    # TODO: extract to a different function
    print("\nDone training! Now testing model...")
    trainer.test(datamodule=dm) # Automatically loads best checkpoint, and tests with test dataloader

    # Generate text from the model
    print("\nGenerating text...")
    input_starting_strings = [
        "<pad>",
        "= valkyria",
        "= = reception ="]

    # Define the device to use
    device = torch.device(device)

    generated_strings = generate_text(
        model=model,
        tokenizer=tokenizer,
        start_string_list=input_starting_strings,
        device=device,
        seq_len=config.seq_len,
        generation_length=100)

    print("Generated strings:")
    for idx, string in enumerate(generated_strings):
        print(f"{idx+1}: {string}\n")

    # TODO: implement this loss value better for grid search.
    return model, trainer.callback_metrics['test_loss'].item()

if __name__ == "__main__":
    args = sys.argv
    config_path =args[1]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    config = Struct(**config)

    start_time = time.time()

    model, test_loss = train_model(config)

    end_time = time.time()

    if config.grid_search_out_file is not None:
        with open(config.grid_search_out_file, "a") as results_file:
            results_file.write(",".join(map(str, [
                config.learning_rate,
                config.embed_dim,
                config.ffn_dim,
                config.heads,
                config.seq_len,
                test_loss,
                end_time - start_time])) + "\n")
    print(f"""\nGRID SEARCH PARAMS: Learning Rate: {config.lr}, 
          Embedding Dimension: {config.embed_dim}, 
          FFN Dimension: {config.ffn_dim}, 
          Heads: {config.heads}, 
          Sequence Length: {config.seq_len}, 
          Test Loss: {test_loss}, 
          Training Time: {end_time - start_time} \n""")


    