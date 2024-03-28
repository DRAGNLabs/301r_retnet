import torch
import torch.nn as nn
from torch import Tensor

from utils import Struct
from pytorch_lightning import LightningModule

class LightningModel(LightningModule):
    
    def __init__(self, config: Struct, hf_config, HFClass):
        """
        Args:
            config (Struct): A Struct object with all configuration fields.
        """
        super().__init__()
        self.learning_rate = config.learning_rate
        self.config = config

        self.model_hf = HFClass(hf_config)

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
        """ Training step, called automatically by PyTorch Lightning. """
        # Unpack batch
        inputs, targets = batch

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
        """ Validation step, called automatically by PyTorch Lightning. """
        # Unpack batch
        inputs, targets = batch

        # Get predictions
        preds = self.model_hf(inputs)

        # Reshape the model predictions for Cross Entropy
        preds = preds.transpose(-2, -1)

        # Calculate loss
        loss = self.loss_fn(preds, targets)

        perplexity = torch.exp(loss)

        self.log(
            name="val_loss",
            value=loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=True)
        
        self.log(
            name="val_perplexity", 
            value=perplexity, 
            prog_bar=True,
            logger=True, 
            on_step=False, 
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=True)

        return loss

    def test_step(self, batch: Tensor, batch_idx: int):
        """ Test step, called automatically by PyTorch Lightning. """
        # Unpack batch
        inputs, targets = batch

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
        """ Configure optimizer and learning rate scheduler. """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma)
        return [optimizer], [lr_scheduler]

    def save_pretrained(self, save_folder: str):
        """ Save model weights and parameters to folder.
        Args:
            save_folder (str): Path to folder to save trained model.
        """
        self.model_hf.save_pretrained(save_folder)