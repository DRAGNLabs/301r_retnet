import json
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from datetime import datetime
from load_data import get_loaders_tokenizer
from math import isclose
from pathlib import Path
import signal
from tabulate import tabulate
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary as model_summary
from torchscale.architecture.config import DecoderConfig, RetNetConfig
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.retnet import RetNetDecoder
from tqdm import tqdm
from transformers import set_seed
from utils import generate_text
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import DataModule

REPO_ROOT_NAME = "301r_retnet"

# Allow torch to run float32 matrix multiplications in lower precision for
# better performance while training if hardware is capable
torch.backends.cuda.matmul.allow_tf32 = True

class RetNetModel(LightningModule):
    """ Create model with RetNet architecture. """
    def __init__(
            self,
            embed_dim: int,
            value_embed_dim: int,
            retention_heads: int,
            ffn_dim: int,
            layers: int,
            dropout: float,
            activation_dropout: float,
            vocab_size: int,
            fsdp: bool,
            max_seq_len: int,
            lr: float):
        """ Use parameters to create corresponding RetNet model.
        Args:
            embed_dim (int): Dimension size of each embedded token.
            value_embed_dim (int): Value embed dimension size.
            retention_heads (int): Number of retention heads in MSR module.
            ffn_dim (int): Hidden layer size of Feed Forward Network (FFN).
            layers (int): Number of retention network layers.
            dropout (float): Probability of an element to be zeroed during
                dropout.
            activation_dropout (float): Probability of an element to be zeroed
                during dropout after activation between FFN layers.
            vocab_size (int): Maximum vocabulary size (number of unique tokens
                in vocabulary.
            fsdp (bool): Whether to shard Module parameters across data parallel
                workers or not (with the FairScale library).
            max_seq_len (int): Size of context window.
        """
        super().__init__()

        # Store hyperparameters
        self.model_params = {
            "embed_dim": embed_dim,
            "value_embed_dim": value_embed_dim,
            "retention_heads": retention_heads,
            "ffn_dim": ffn_dim,
            "layers": layers,
            "dropout": dropout,
            "activation_dropout": activation_dropout,
            "vocab_size": vocab_size,
            "fsdp": fsdp,
            "max_seq_len": max_seq_len,
            "lr": lr}

        # Create RetNet configuration
        config = RetNetConfig(
            decoder_embed_dim=embed_dim,
            decoder_value_embed_dim=value_embed_dim,
            decoder_retention_heads=retention_heads,
            decoder_ffn_embed_dim=ffn_dim,
            decoder_layers=layers,
            dropout=dropout,
            activation_dropout=activation_dropout,
            vocab_size=vocab_size,
            fsdp=fsdp)

        # Create embeddings with index 0 representing padding
        text_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0)

        self.decoder_stack = RetNetDecoder(config, embed_tokens=text_embeddings)

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
        preds, _ = self.decoder_stack(x)
        return preds
    
    def training_step(self, batch, batch_idx):
        # TODO: this may not work, need to check
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        preds, _ = self.decoder_stack(inputs)

        # Reshape the model predictions for Cross Entropy
        preds = preds.transpose(-2, -1)

        # Calculate loss
        loss = self.loss_fn(preds, targets)
        train_total_loss += loss * len(inputs)
        train_total_samples += len(inputs)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Put validation inputs and targets on device
        val_inputs = batch[:, :-1]
        val_targets = batch[:, 1:]

        # Get validation predictions
        val_predictions = model(val_inputs)

        # Reshape the model predictions for Cross Entropy
        val_predictions = val_predictions.transpose(-2, -1)

        # Calculate validation loss
        val_loss = self.loss_fn(val_predictions, val_targets)

        return val_loss

    def get_params(self) -> dict:
        """ Get model parameters dictionary. """
        return self.model_params
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.decoder_stack.parameters(), lr=self.model_params["lr"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma)
        return [optimizer], [lr_scheduler]


class TransformerModel(LightningModule):
    def __init__(
            self,
            embed_dim: int,
            value_embed_dim: int,
            attention_heads: int,
            ffn_dim: int,
            layers: int,
            dropout: float,
            activation_dropout: float,
            vocab_size: int,
            fsdp: bool,
            max_seq_len: int):
        """ Use parameters to create corresponding Transformer model.
        Args:
            embed_dim (int): Dimension size of each embedded token.
            value_embed_dim (int): Value embed dimension size.
            attention_heads (int): Number of attention heads in MHA module.
            ffn_dim (int): Hidden layer size of Feed Forward Network (FFN).
            layers (int): Number of retention network layers.
            dropout (float): Probability of an element to be zeroed during
                dropout.
            activation_dropout (float): Probability of an element to be zeroed
                during dropout after activation between FFN layers.
            vocab_size (int): Maximum vocabulary size (number of unique tokens
                in vocabulary.
            fsdp (bool): Whether to shard Module parameters across data parallel
                workers or not (with the FairScale library).
            max_seq_len (int): Size of context window.
        """
        super().__init__()

        # Store hyperparameters
        self.model_params = {
            "embed_dim": embed_dim,
            "value_embed_dim": value_embed_dim,
            "attention_heads": attention_heads,
            "ffn_dim": ffn_dim,
            "layers": layers,
            "dropout": dropout,
            "activation_dropout": activation_dropout,
            "vocab_size": vocab_size,
            "fsdp": fsdp,
            "max_seq_len": max_seq_len}

        # Create Transformer Decoder configuration
        config = DecoderConfig(
            decoder_embed_dim=embed_dim,
            decoder_value_embed_dim=value_embed_dim,
            decoder_attention_heads=attention_heads,
            decoder_ffn_embed_dim=ffn_dim,
            decoder_layers=layers,
            dropout=dropout,
            activation_dropout=activation_dropout,
            vocab_size=vocab_size,
            fsdp=fsdp)

        # Create embeddings with index 0 representing padding
        text_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0)

        self.decoder_stack = Decoder(config, embed_tokens=text_embeddings)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Long tensor of dimensions: (batch size, sequence
                length).

        Returns:
            A tensor of dimensions: (batch size, sequence length, vocabulary
                size).
        """
        preds, _ = self.decoder_stack(x)
        return preds
    
    def training_step(self, batch, batch_idx):
        # TODO: this may not work, need to check
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        preds, _ = self.decoder_stack(inputs)

        # Reshape the model predictions for Cross Entropy
        preds = preds.transpose(-2, -1)

        # Calculate loss
        loss = self.loss_fn(preds, targets)
        train_total_loss += loss * len(inputs)
        train_total_samples += len(inputs)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Put validation inputs and targets on device
        val_inputs = batch[:, :-1]
        val_targets = batch[:, 1:]

        # Get validation predictions
        val_predictions = model(val_inputs)

        # Reshape the model predictions for Cross Entropy
        val_predictions = val_predictions.transpose(-2, -1)

        # Calculate validation loss
        val_loss = self.loss_fn(val_predictions, val_targets)

        return val_loss

    def get_params(self) -> dict:
        """ Get model parameters dictionary. """
        return self.model_params
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.decoder_stack.parameters(), lr=self.model_params["lr"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma)
        return [optimizer], [lr_scheduler]


if __name__ == "__main__":
    # Initialize, setup, and parse the argument parser
    parser = ArgumentParser(
        prog="Model Trainer",
        description="Used to train comparable RetNet, Transformer models.")

    parser.add_argument("-a", "--activation-dropout", type=float, default=0.0,
        help="Probability of element to be zeroed in dropout layer after " + \
            "activation between FFN layers.")
    parser.add_argument("-b", "--batch-size", type=int, default=32,
        help="Batch size.")
    parser.add_argument("-c", "--checkpoints", action="store_true",
        default=False, help="Save model checkpoints while training.")
    parser.add_argument("--dataset-feature", type=str, default="text",
        help="Hugging Face dataset feature/column to use.")
    parser.add_argument("--dataset-name", type=str, default="wikitext",
        help="Hugging Face dataset name. Should also set --dataset-subset.")
    parser.add_argument("--dataset-subset", type=str, default="wikitext-2-v1",
        help="Subset/config to use for Hugging Face dataset.")
    parser.add_argument("--device", type=str, default="cuda",
        help="Device to use (ex: 'cpu', 'cuda', or 'cuda:0').")
    parser.add_argument("-d", "--dropout", type=float, default=0.1,
        help="Probability of element to be zeroed in dropout layer.")
    parser.add_argument("-e", "--embed-dim", type=int, default=768,
        help="Embedding dimension size of each token.")
    parser.add_argument("--epochs", type=int, default=10,
        help="Number of epochs to train for.")
    parser.add_argument("-f", "--ffn-dim", type=int, default=1280,
        help="FFN hidden layer size.")
    parser.add_argument("--fsdp", action="store_true", default=False,
        help="Module parameters sharded across data parallel workers.")
    parser.add_argument("-l", "--layers", type=int, default=12,
        help="Number of stacked layers in model.")
    parser.add_argument("--lr", type=float, required=True,
        help="Learning rate of model to train.")
    parser.add_argument("-m", "--model", required=True,
        choices=["retnet", "transformer"],
        help="Name of model architecture to train.")
    parser.add_argument("-n", "--heads", type=int, default=3,
        help="Number of heads. Head architecture changes based on model.")
    parser.add_argument("-r", "--rand-seed", type=int, default=None,
        help="Random seed to use, allowing more reproducible results.")
    parser.add_argument("-s", "--seq-len", type=int, default=512,
        help="Sequence length (context window size).")
    parser.add_argument("--splits", type=float, nargs=3,
        default=[0.7, 0.2, 0.1],
        help="Space-separated decimal splits of train, validation, and " + \
            "test datasets. (Ex: '0.7 0.2 0.1')")
    parser.add_argument("--val-freq", type=int, default=3,
        help="Number of times to run validation per epoch during training.")
    parser.add_argument("--value-embed-dim", type=int, default=1280,
        help="Value embed dimension size.")
    parser.add_argument("--vocab-size", type=int, required=True,
        help="Maximum number of unique tokens in vocabulary.")

    args = parser.parse_args()

    # Test that the head dimension will be an even, whole number
    assert args.embed_dim % (args.heads * 2) == 0, \
        "Head Dimension must be even to perform Rotary Position Embedding " + \
        f"({args.embed_dim} / {args.heads} = {args.embed_dim / args.heads} " + \
        "-- not an even, whole number)! Try changing the Embedding " + \
        "Dimension or number of heads."

    # Test that the value embedding dimension is divisible by number of heads
    assert args.value_embed_dim % args.heads == 0, \
        "Value Embed Dimension not divisible by number of heads " + \
        f"({args.value_embed_dim} % {args.heads} != 0)!"

    # Test the dataset splits add up to 1, using isclose for rounding errors
    assert isclose(sum(args.splits), 1), \
        "The dataset splits for the training, validation, and testing " + \
        f"datasets must sum up to 1 ({' + '.join(map(str, args.splits))} != 1)!"

    # Set random seeds for torch, numpy, random, etc. with transformers library
    if args.rand_seed is not None:
        set_seed(args.rand_seed)

    # Create requested model
    if args.model == "retnet":
        model = RetNetModel(
            embed_dim=args.embed_dim,
            value_embed_dim=args.value_embed_dim,
            retention_heads=args.heads,
            ffn_dim=args.ffn_dim,
            layers=args.layers,
            dropout=args.dropout,
            activation_dropout=args.activation_dropout,
            vocab_size=args.vocab_size,
            fsdp=args.fsdp,
            max_seq_len=args.seq_len)
    elif args.model == "transformer":
        model = TransformerModel(
            embed_dim=args.embed_dim,
            value_embed_dim=args.value_embed_dim,
            attention_heads=args.heads,
            ffn_dim=args.ffn_dim,
            layers=args.layers,
            dropout=args.dropout,
            activation_dropout=args.activation_dropout,
            vocab_size=args.vocab_size,
            fsdp=args.fsdp,
            max_seq_len=args.seq_len)

    # Print all arguments for recordkeeping
    print("Arguments:")
    arg_table = []
    row = []
    for i, arg in enumerate(vars(args)):
        row.append(f"{arg}: {getattr(args, arg)}")
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
        input_data=torch.ones(1, args.seq_len).long()).total_params

    # Get path of repository root folder
    repo_root_dir = Path(__file__)
    while REPO_ROOT_NAME not in repo_root_dir.name:
        repo_root_dir = repo_root_dir.parent

    # Create unique label for model (timestamp, model type, parameter count)
    model_label = f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}_" + \
        f"{args.model}_{total_params}"

    # Initialize model weights folders
    save_folder = repo_root_dir / "weights" / model_label
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving weights in {save_folder}")

    # Save all the variables in args as JSON inside folder
    arg_dict = vars(args)
    json_string = json.dump(
        obj=arg_dict,
        fp=open(save_folder / "model_args.json", "w"),
        indent=4)

    # Create SummaryWriter to record logs for TensorBoard
    # TODO: implement with lightning
    writer = SummaryWriter(log_dir=repo_root_dir / "logs" / model_label)

    # Print estimated loss if it hasn't learned anything
    print("\nEstimated Loss if guessing:")
    print(f"-log(1 / {args.vocab_size}) = " + \
        f"{-torch.log(torch.tensor(1 / args.vocab_size))}")

    # Get DataLoaders and trained Tokenizer
    print(f"\nNow retrieving '{args.dataset_name}' and training tokenizer...")
    dm = DataModule(args.data_dir, args.batch_size)

    model = torch.compile(model)

    # Implement callbacks
    model_checkpoint = ModelCheckpoint(
        dirpath=save_folder,
        filename='epoch_{epoch}_validation_{num_val_runs}', # TODO: where are we getting num val runs?
        save_top_k=args.save_top_k,
        monitor='val_loss',
        mode='max')
    
    # TODO: make sure these args exist
    trainer = Trainer(
        default_root_dir=args.default_root_dir, # main directory for run
        accelerator=args.accelerator, # gpu or cpu
        num_nodes=args.num_nodes,
        devices=args.devices,
        strategy="ddp",
        max_epochs=args.num_epochs,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        sync_batchnorm=True,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
        callbacks=[model_checkpoint],
        logger=logger
        )
    
    trainer.fit(model, datamodule=dm)

    print("\nDone training! Now testing model...")
    trainer.test() # Automatically loads best checkpoint, and tests with test dataloader

    # Save hyperparameters and metrics in logs
    writer.add_hparams(
        hparam_dict=model.get_params(),
        metric_dict={
            "Loss/train": avg_train_loss,
            "Loss/validation": avg_val_loss,
            "Loss/test": avg_loss})

    # Close SummaryWriter
    writer.close()

    # Generate text from the model
    print("\nGenerating text...")
    input_starting_strings = [
        "<pad>",
        "= valkyria",
        "= = reception ="]

    # Define the device to use
    device = torch.device(args.device)

    generated_strings = generate_text(
        model=model,
        tokenizer=tokenizer,
        start_string_list=input_starting_strings,
        device=device,
        seq_len=args.seq_len,
        generation_length=100)

    print("Generated strings:")
    for idx, string in enumerate(generated_strings):
        print(f"{idx+1}: {string}\n")
