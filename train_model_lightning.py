import os
import json
import time
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
from transformers import set_seed, PreTrainedTokenizerFast
from utils import generate_text
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import DataModule
from hugging_face_model import RetNetModelHF, TransformerModelHF

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
            fsdp=fsdp,
            max_seq_len=max_seq_len,
            lr=lr)

        self.model_hf = RetNetModelHF(config)

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
        optimizer = torch.optim.Adam(self.model_hf.decoder_stack.parameters(), lr=self.model_params["lr"])
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma) # TODO: Implement this
        return [optimizer]#, [lr_scheduler]
    
    def save_pretrained(self, save_folder: str):
        """ Save model weights and parameters to folder. """
        self.model_hf.save_pretrained(save_folder)

    # @staticmethod
    # def from_pretrained(load_folder: str):
    #     """ Load model hugging face weights and parameters from folder"""

    #     model = RetNetModel(blah blah)
    #     model.model_hf = RetNetModelHF.from_pretrained(load_folder)
    #     return model

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
            max_seq_len: int,
            lr: float):
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
            "max_seq_len": max_seq_len,
            "lr": lr}

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
        return self.model_params
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model_hf.parameters(), lr=self.model_params["lr"])
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma)
        return [optimizer]#, [lr_scheduler]
    
    def save_pretrained(self, save_folder: str):
        """ Save model weights and parameters to folder. """
        self.model_hf.save_pretrained(save_folder)

def train_model(activation_dropout=0.0, 
                batch_size=8, 
                checkpoints=False, 
                data_dir=None,
                dataset_dir=None,
                dataset_feature=None, 
                dataset_name="wikitext", 
                dataset_subset="wikitext-2-v1", 
                device="cuda",
                dropout=0.1, 
                embed_dim=76, 
                epochs=1, 
                ffn_dim=12, 
                fsdp=False, 
                heads=4, 
                layers=2, 
                lr=0.001, 
                model_type="retnet", 
                rand_seed=None, 
                repo_root_dir=None,
                seq_len=128, 
                splits=[0.7, 0.2, 0.1], 
                tboard_dir=None, 
                val_freq=1, 
                value_embed_dim=12, 
                vocab_size=4000,
                num_devices=1,
                tokenizer_folder=None,
                train_data=None,
                validation_data=None,
                test_data=None):
    arg_dict = locals()
    print(arg_dict)

    # Test that the head dimension will be an even, whole number
    assert embed_dim % (heads * 2) == 0, \
        "Head Dimension must be even to perform Rotary Position Embedding " + \
        f"({embed_dim} / {heads} = {embed_dim / heads} " + \
        "-- not an even, whole number)! Try changing the Embedding " + \
        "Dimension or number of heads."

    # Test that the value embedding dimension is divisible by number of heads
    assert value_embed_dim % heads == 0, \
        "Value Embed Dimension not divisible by number of heads " + \
        f"({value_embed_dim} % {heads} != 0)!"

    # Test the dataset splits add up to 1, using isclose for rounding errors
    assert isclose(sum(splits), 1), \
        "The dataset splits for the training, validation, and testing " + \
        f"datasets must sum up to 1 ({' + '.join(map(str, splits))} != 1)!"

    # Set random seeds for torch, numpy, random, etc. with transformers library
    if rand_seed is not None:
        set_seed(rand_seed)

    # Create requested model
    if model_type == "retnet":
        model = RetNetModel(
            embed_dim=embed_dim,
            value_embed_dim=value_embed_dim,
            retention_heads=heads,
            ffn_dim=ffn_dim,
            layers=layers,
            dropout=dropout,
            activation_dropout=activation_dropout,
            vocab_size=vocab_size,
            fsdp=fsdp,
            max_seq_len=seq_len,
            lr=lr)
    elif model_type == "transformer":
        model = TransformerModel(
            embed_dim=embed_dim,
            value_embed_dim=value_embed_dim,
            attention_heads=heads,
            ffn_dim=ffn_dim,
            layers=layers,
            dropout=dropout,
            activation_dropout=activation_dropout,
            vocab_size=vocab_size,
            fsdp=fsdp,
            max_seq_len=seq_len,
            lr=lr)

    # Print all arguments for recordkeeping
    """print("Arguments:")
    arg_table = []
    row = []
    for i, arg in enumerate(vars(args)):
        row.append(f"{arg}: {getattr(args, arg)}")
        if (i + 1) % 4 == 0:
            arg_table.append(row)
            row = []
    if row:
        arg_table.append(row)
    print(tabulate(arg_table, tablefmt="grid"))"""

    # Print model info
    print("\nModel Summary:")
    total_params = model_summary(
        model,
        input_data=torch.ones(1, seq_len).long()).total_params
    
    # Create unique label for model (timestamp, model type, parameter count)
    model_label = f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}_" + \
        f"{model_type}_{total_params}"

    # Initialize model weights folders
    save_folder = Path(data_dir) / "weights" / model_label
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving weights in {save_folder}")

    # Save all the variables in args as JSON inside folder
    """arg_dict = vars(args)
    json_string = json.dump(
        obj=arg_dict,
        fp=open(save_folder / "model_args.json", "w"),
        indent=4)"""

    # Print estimated loss if it hasn't learned anything
    print("\nEstimated Loss if guessing:")
    print(f"-log(1 / {vocab_size}) = " + \
        f"{-torch.log(torch.tensor(1 / vocab_size))}")

    # Get DataLoaders and trained Tokenizer
    # Get Tokenizer from local directory
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_folder)

    # Loads Tokenized data
    # train_tokenized_dataset_path = str(Path(dataset_dir) / dataset_name / "tokenized" / "train.parquet")
    # test_tokenized_dataset_path = str(Path(dataset_dir) / dataset_name / "tokenized" / "test.parquet")
    # validation_tokenized_dataset_path = str(Path(dataset_dir) / dataset_name / "tokenized" / "validation.parquet")

    print(f"\nNow retrieving '{dataset_name}' and tokenizer...")
    dm = DataModule(train_data, 
                    test_data,
                    validation_data,
                    batch_size)

    #model = torch.compile(model) #TODO: this doesn't work with lightning, says something about logging in validation twice: need to use a different version of python?

    # Implement callbacks
    model_checkpoint = CustomCheckpoint(
        dirpath=save_folder,
        filename='epoch_{epoch}_validation_{num_val_runs}', # TODO: where are we getting num val runs?
        save_top_k=3, # TODO: implement this argument
        monitor='val_loss',
        mode='max')
    
    trainer = Trainer(
        default_root_dir=data_dir, # main directory for run
        accelerator='gpu', # gpu or cpu
        num_nodes=1, # TODO: implement this as an argument
        devices=num_devices,
        strategy="ddp",
        max_epochs=epochs,
        accumulate_grad_batches=1, #TODO: implement this argument
        sync_batchnorm=True,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
        callbacks=[model_checkpoint],
        #val_check_interval=val_freq #TODO: need to set this properly-> default is 1? But doesn't make sense.
        )
    
    trainer.fit(model, datamodule=dm)

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
        seq_len=seq_len,
        generation_length=100)

    print("Generated strings:")
    for idx, string in enumerate(generated_strings):
        print(f"{idx+1}: {string}\n")

    # TODO: implement this loss value better for grid search.
    return model, trainer.callback_metrics['test_loss'].item()

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
    parser.add_argument("--data-dir", type=str, required=True,
        help="Path to directory where all data except datasets are saved.")
    parser.add_argument("--dataset-dir", type=str, required=True,
        help="Path to directory in which Hugging Face datasets are downloaded.")
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
    parser.add_argument("--tboard-dir", type=str, default=None,
        help="Path to directory to save TensorBoard logs in.")
    parser.add_argument("--val-freq", type=int, default=3,
        help="Number of times to run validation per epoch during training.")
    parser.add_argument("--value-embed-dim", type=int, default=1280,
        help="Value embed dimension size.")
    parser.add_argument("--vocab-size", type=int, required=True,
        help="Maximum number of unique tokens in vocabulary.")
    parser.add_argument("--tokenizer-folder", type= str, required=True,
        help="Path to the file where the tokenizer will be saved")
    parser.add_argument("--num-devices", type= str, required=True,
        help="Number of gpus to train on")
    parser.add_argument("--train-data", type= str, required=True,
        help="Direct path to tokenized train data")
    parser.add_argument("--validation-data", type= str, required=True,
        help="Direct path to tokenized validation data")
    parser.add_argument("--test-data", type= str, required=True,
        help="Direct path to tokenized test data")
    parser.add_argument("--grid-search-out-file", type=str, default=None)
    

    args = parser.parse_args()

    start_time = time.time()

    model, test_loss = train_model(
        activation_dropout=args.activation_dropout, 
        batch_size=args.batch_size, 
        checkpoints=args.checkpoints, 
        data_dir=args.data_dir,
        dataset_dir=args.dataset_dir,
        dataset_feature=args.dataset_feature,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        device=args.device, 
        dropout=args.dropout, 
        embed_dim=args.embed_dim, 
        epochs=args.epochs, 
        ffn_dim=args.ffn_dim, 
        fsdp=args.fsdp, 
        heads=args.heads, 
        layers=args.layers, 
        lr=args.lr, 
        model_type=args.model, 
        rand_seed=args.rand_seed, 
        seq_len=args.seq_len, 
        splits=args.splits,
        tboard_dir=args.tboard_dir,
        val_freq=args.val_freq, 
        value_embed_dim=args.value_embed_dim, 
        vocab_size=args.vocab_size,
        num_devices=args.num_devices,
        tokenizer_folder=args.tokenizer_folder,
        train_data = args.train_data,
        validation_data = args.validation_data,
        test_data = args.test_data
    )

    end_time = time.time()

    if args.grid_search_out_file is not None:
        with open(args.grid_search_out_file, "a") as results_file:
            results_file.write(",".join(map(str, [
                args.lr,
                args.embed_dim,
                args.batch_size,
                args.model,
                test_loss,
                end_time - start_time])) + "\n")
    print(f"GRID SEARCH PARAMS: Learning Rate: {args.lr}, Embedding Dimension: {args.embed_dim}, Batch Size: {args.batch_size}, Model Type: {args.model} Avg Loss: {test_loss}, Training Time: {end_time - start_time} \n")


    