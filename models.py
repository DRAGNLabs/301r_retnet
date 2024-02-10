from pytorch_lightning import LightningModule
import torch
from torch import Tensor
import torch.nn as nn
from torchscale.architecture.config import RetNetConfig, DecoderConfig
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.retnet import RetNetDecoder
from transformers import PreTrainedModel
from typing import Optional, Union
from utils import Struct

class RetNetModel(LightningModule):
    """ 
    Create model with RetNet architecture.

    This is a LightningModule that wraps around a 
    HuggingFace class, containing the RetNet architecture.
    """
    def __init__(self, config: Struct):
        super().__init__()
        self.learning_rate = config.learning_rate
        self.gamma = config.gamma

        # Create RetNet configuration for HuggingFace model
        hf_config = RetNetConfig(
            decoder_embed_dim=config.embed_dim,
            decoder_value_embed_dim=config.value_embed_dim,
            decoder_retention_heads=config.heads,
            decoder_ffn_embed_dim=config.ffn_dim,
            decoder_layers=config.layers,
            dropout=config.dropout,
            activation_dropout=config.activation_dropout,
            vocab_size=config.vocab_size,
            max_seq_len=config.seq_len,
            lr=config.learning_rate)

        self.model_hf = RetNetModelHF(hf_config)

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

        self.save_hyperparameters(self.model_hf.get_params())

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

    def training_step(self, batch: Tensor, batch_idx: int):
        """
        Training step, called automatically by PyTorch Lightning.
        """
        # Unpack batch
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        # Get predictions
        preds = self.model_hf(inputs)

        # Reshape the model predictions for Cross Entropy
        preds = preds.transpose(-2, -1)

        # Calculate loss
        loss = self.loss_fn(preds, targets)

        self.log(
            name="train_loss",
            value=loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=True)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        """
        Validation step, called automatically by PyTorch Lightning.
        """
        # Unpack batch
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        # Get predictions
        preds = self.model_hf(inputs)

        # Reshape the model predictions for Cross Entropy
        preds = preds.transpose(-2, -1)

        # Calculate loss
        loss = self.loss_fn(preds, targets)

        self.log(
            name="val_loss",
            value=loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=True)

        return loss

    def test_step(self, batch: Tensor, batch_idx: int):
        """
        Test step, called automatically by PyTorch Lightning.
        """
        # Unpack batch
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        # Get predictions
        preds = self.model_hf(inputs)

        # Reshape the model predictions for Cross Entropy
        preds = preds.transpose(-2, -1)

        # Calculate loss
        loss = self.loss_fn(preds, targets)

        self.log(
            name="test_loss",
            value=loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=True)

        return loss

    def get_params(self) -> dict:
        """ Get model parameters dictionary. """
        return self.model_hf.get_params()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler for Lightning to use."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.gamma)
        return [optimizer], [lr_scheduler]

    def save_pretrained(self, save_folder: str):
        """ Save model weights and parameters to folder. """
        self.model_hf.save_pretrained(save_folder)


class TransformerModel(LightningModule):
    """
    Create model with Transformer architecture.

    This is a LightningModule that wraps around a
    HuggingFace class, containing the Transformer architecture.
    """
    def __init__(self, config: Struct):
        super().__init__()
        self.learning_rate = config.learning_rate
        self.gamma = config.gamma

        # Create Transformer Decoder configuration for HuggingFace model
        config = DecoderConfig(
            decoder_embed_dim=config.embed_dim,
            decoder_value_embed_dim=config.value_embed_dim,
            decoder_attention_heads=config.heads,
            decoder_ffn_embed_dim=config.ffn_dim,
            decoder_layers=config.layers,
            dropout=config.dropout,
            activation_dropout=config.activation_dropout,
            vocab_size=config.vocab_size)

        self.model_hf = TransformerModelHF(config)

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

        self.save_hyperparameters(self.model_hf.get_params())

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

    def training_step(self, batch: Tensor, batch_idx: int):
        """
        Training step, called automatically by PyTorch Lightning.
        """
        # Unpack batch
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        preds = self.model_hf(inputs)

        # Reshape the model predictions for Cross Entropy
        preds = preds.transpose(-2, -1)

        # Calculate loss
        loss = self.loss_fn(preds, targets)

        self.log(
            name="train_loss",
            value=loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=True)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        """
        Validation step, called automatically by PyTorch Lightning.
        """
        # Unpack batch
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        preds = self.model_hf(inputs)

        # Reshape the model predictions for Cross Entropy
        preds = preds.transpose(-2, -1)

        # Calculate loss
        loss = self.loss_fn(preds, targets)

        self.log(
            name="val_loss",
            value=loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=True)

        return loss

    def test_step(self, batch: Tensor, batch_idx: int):
        """
        Test step, called automatically by PyTorch Lightning.
        """
        # Unpack batch
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        preds = self.model_hf(inputs)

        # Reshape the model predictions for Cross Entropy
        preds = preds.transpose(-2, -1)

        # Calculate loss
        loss = self.loss_fn(preds, targets)

        self.log(
            name="test_loss",
            value=loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=True)

        return loss

    def get_params(self) -> dict:
        """ Get model parameters dictionary. """
        return self.model_hf.get_params()

    def configure_optimizers(self):
        """ Configure optimizer and learning rate scheduler for Lightning to use. """
        optimizer = torch.optim.Adam(
            self.model_hf.decoder_stack.parameters(),
            lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.gamma)
        return [optimizer], [lr_scheduler]

    def save_pretrained(self, save_folder: str):
        """ Save model weights and parameters to folder. """
        self.model_hf.save_pretrained(save_folder)


class RetNetModelHF(PreTrainedModel):
    """ Create model with RetNet architecture. """
    def __init__(self, config: Optional[RetNetConfig]=None):
        """ Use configuration object to create corresponding RetNet model.
        Args:
            config (RetNetConfig): A RetNet configuration object.
        """
        # Create RetNet configuration
        if config is None:
            self.config = RetNetConfig()
        elif isinstance(config, RetNetConfig):
            self.config = config
        else:
            raise ValueError("If given, config must be a RetNetConfig object.")

        super().__init__(self.config)

        # Create embeddings with index 0 representing padding
        text_embeddings = nn.Embedding(
            num_embeddings=int(self.config.vocab_size),
            embedding_dim=int(self.config.decoder_embed_dim),
            padding_idx=0)

        self.decoder_stack = RetNetDecoder(
            self.config,
            embed_tokens=text_embeddings)

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
    """ Create model with Transformer architecture. """
    def __init__(self, config: Optional[DecoderConfig]=None):
        """ Use configuration object to create corresponding Transformer model.
        Args:
            config (DecoderConfig): A Decoder configuration object.
        """
        # Create Transformer configuration
        if not config:
            self.config = DecoderConfig()
        elif isinstance(config, DecoderConfig):
            self.config = config
        else:
            raise ValueError("If given, config must be a DecoderConfig object.")

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
