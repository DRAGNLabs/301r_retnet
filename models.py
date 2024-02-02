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
from dataset import DataModule
from torchscale.architecture.config import RetNetConfig, DecoderConfig
from utils import Struct

# TODO: clean up all these imports
import torch.nn as nn

from torch import Tensor
from typing import Optional, Union
from torchscale.architecture.config import DecoderConfig, RetNetConfig
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.retnet import RetNetDecoder
from transformers import PreTrainedModel


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

class RetNetModelHF(PreTrainedModel):
    """ Create model with RetNet architecture. """
    config_class = RetNetConfig

    def __init__(
            self,
            config: Optional[Union[RetNetConfig, str]] = None):
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
            config (str): Path to RetNet configuration file.
        """

        # Create RetNet configuration
        if not config:
            self.config = RetNetConfig()
        elif isinstance(config, str):
            self.config = RetNetConfig.from_pretrained(config)
        elif isinstance(config, RetNetConfig):
            self.config = config
        else:
            raise ValueError("Config must be str or RetNetConfig object.")

        super().__init__(self.config)

        # Create embeddings with index 0 representing padding
        text_embeddings = nn.Embedding(
            num_embeddings=int(self.config.vocab_size),
            embedding_dim=int(self.config.decoder_embed_dim),
            padding_idx=0)

        self.decoder_stack = RetNetDecoder(self.config, embed_tokens=text_embeddings)

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

    def get_params(self) -> dict:
        """ Get model parameters dictionary. """
        allowed_types = (int, float, str, bool, Tensor)
        hparams = self.config.to_dict()
        for key, value in hparams.items():
            if not isinstance(value, allowed_types):
                hparams[key] = str(value)
        return hparams


class TransformerModelHF(PreTrainedModel):
    config_class = DecoderConfig

    def __init__(
            self, config: Optional[Union[DecoderConfig, str]] = None):
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

        # Create Transformer configuration
        if not config:
            self.config = DecoderConfig()
        elif isinstance(config, str):
            self.config = DecoderConfig.from_pretrained(config)
        elif isinstance(config, DecoderConfig):
            self.config = config
        else:
            raise ValueError("Config must be str or DecoderConfig object.")

        super().__init__(self.config)

        # Create embeddings with index 0 representing padding
        text_embeddings = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.decoder_embed_dim,
            padding_idx=0)

        self.decoder_stack = Decoder(self.config, embed_tokens=text_embeddings)

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

    def get_params(self) -> dict:
        """ Get model parameters dictionary. """
        allowed_types = (int, float, str, bool, Tensor)
        hparams = self.config.to_dict()
        for key, value in hparams.items():
            if not isinstance(value, allowed_types):
                hparams[key] = str(value)
        return hparams
