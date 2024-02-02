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
from hugging_face_model import RetNetModelHF, TransformerModelHF
from dataset import DataModule
from torchscale.architecture.config import RetNetConfig, DecoderConfig
from utils import Struct



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