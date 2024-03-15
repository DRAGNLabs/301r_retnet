import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from torch import Tensor
from torchscale.architecture.config import RetNetConfig, DecoderConfig, PerformerConfig
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.retnet import RetNetDecoder
from torchscale.architecture.performer import PerformerDecoder
from transformers import PreTrainedModel
from typing import Optional, Union
from utils import Struct

class RetNetModel(LightningModule):
    """ Create model with RetNet architecture.

    This is a LightningModule that wraps around a HuggingFace class containing
    the RetNet architecture.
    """
    def __init__(self, config: Struct):
        """
        Args:
            config (Struct): A Struct object with all configuration fields.
        """
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.gamma)
        return [optimizer], [lr_scheduler]

    def save_pretrained(self, save_folder: str):
        """ Save model weights and parameters to folder.
        Args:
            save_folder (str): Path to folder to save trained model.
        """
        self.model_hf.save_pretrained(save_folder)


class TransformerModel(LightningModule):
    """ Create model with Transformer architecture.

    This is a LightningModule that wraps around a HuggingFace class containing
    the Transformer architecture.
    """
    def __init__(self, config: Struct):
        """
        Args:
            config (Struct): A Struct object with all configuration fields.
        """
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
        """ Training step, called automatically by PyTorch Lightning. """
        # Unpack batch
        inputs, targets = batch

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
        """ Test step, called automatically by PyTorch Lightning. """
        # Unpack batch
        # inputs = batch[:, :-1]
        # targets = batch[:, 1:]
        inputs, targets = batch

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
        optimizer = torch.optim.Adam(
            self.model_hf.decoder_stack.parameters(),
            lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.gamma)
        return [optimizer], [lr_scheduler]

    def save_pretrained(self, save_folder: str):
        """ Save model weights and parameters to folder.
        Args:
            save_folder (str): Path to folder to save trained model.
        """
        self.model_hf.save_pretrained(save_folder)


class PerformerModel(LightningModule):
    """
    Create a model with Performer architecture within the PyTorch Lightning framework.

    This LightningModule wraps around a class containing the Performer architecture, facilitating training, validation, and testing
    within a structured Lightning workflow. The module leverages a configuration structure for easy adjustments and Hugging Face compatibility for model serialization.
    """
    def __init__(self, config: Struct):
        """
        Initializes the PerformerModel with the provided configuration.

        Args:
            config (Struct): A Struct object encapsulating all the necessary configuration parameters.
        """
        super().__init__()
        self.learning_rate = config.learning_rate
        self.gamma = config.gamma  # Used for learning rate decay.

        # Convert the simple Struct config to the specific Performer configuration required by the Hugging Face model.
        performer_config = PerformerConfig(
            embed_dim=config.embed_dim,
            heads=config.heads,
            depth=config.depth,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            activation_dropout=config.activation_dropout,
            vocab_size=config.vocab_size,
            max_seq_len=config.seq_len,
            lr=config.learning_rate
        )

        # Initialize the Performer model using the defined configuration.
        self.model_hf = PerformerModelHF(performer_config)
        # Cross-entropy loss for training and evaluation.
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        # Store model's hyperparameters for checkpointing.
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Performer model.

        Args:
            x (Tensor): Input tensor of token IDs, shaped (batch_size, sequence_length).

        Returns:
            Tensor: Model predictions, shaped (batch_size, sequence_length, vocab_size).
        """
        preds = self.model_hf(x)
        return preds

    def training_step(self, batch: Tensor, batch_idx: int):
        """
        Performs a training step using a single batch of data.

        Args:
            batch (Tensor): Batch of data provided by the DataLoader.
            batch_idx (int): Index of the batch in the current epoch.

        Returns:
            Tensor: Computed loss for the batch.
        """
        inputs, targets = batch[:, :-1], batch[:, 1:]
        preds = self(inputs)
        preds = preds.transpose(-2, -1)  # Reshape for compatibility with CrossEntropyLoss.
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        """
        Performs a validation step using a single batch of data.

        Args:
            batch (Tensor): Batch of data provided by the DataLoader.
            batch_idx (int): Index of the batch in the validation set.

        Returns:
            Tensor: Computed loss for the batch.
        """
        inputs, targets = batch[:, :-1], batch[:, 1:]
        preds = self(inputs)
        preds = preds.transpose(-2, -1)  # Reshape for compatibility with CrossEntropyLoss.
        loss = self.loss_fn(preds, targets)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch: Tensor, batch_idx: int):
        """
        Performs a test step using a single batch of data.

        Args:
            batch (Tensor): Batch of data provided by the DataLoader.
            batch_idx (int): Index of the batch in the test set.

        Returns:
            Tensor: Computed loss for the batch.
        """
        inputs, targets = batch[:, :-1], batch[:, 1:]
        preds = self(inputs)
        preds = preds.transpose(-2, -1)  # Reshape for compatibility with CrossEntropyLoss.
        loss = self.loss_fn(preds, targets)
        self.log("test_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler used for training.

        Returns:
            List: List containing the optimizer and the learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        return [optimizer], [lr_scheduler]

    def save_pretrained(self, save_folder: str):
        """
        Saves the model weights and configuration to the specified folder.

        Args:
            save_folder (str): The path to the folder where the model should be saved.
        """
        self.model_hf.save_pretrained(save_folder)


class RetNetModelHF(PreTrainedModel):
    """ Create model with RetNet architecture. """
    config_class = RetNetConfig

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
    config_class = DecoderConfig

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


class PerformerModelHF(PreTrainedModel):
    """
    Create model with Performer architecture. This class extends the Hugging Face PreTrainedModel,
    allowing it to be used seamlessly within the Hugging Face ecosystem, including serialization,
    tokenization, and integration with Hugging Face's training and inference pipelines.
    """
    config_class = PerformerConfig

    def __init__(self, config: Optional[PerformerConfig] = None):
        """
        Initializes the Performer model with the given configuration. If no configuration is provided,
        initializes with default config settings.

        Args:
            config (Optional[PerformerConfig]): Configuration object specifying model architecture parameters.
        """
        # Validate or set default configuration for the Performer model
        if config is None:
            self.config = PerformerConfig()
        elif isinstance(config, PerformerConfig):
            self.config = config
        else:
            raise ValueError("config must be of type PerformerConfig if provided")

        super().__init__(self.config)

        # Create embeddings layer with padding index set to 0
        self.embeddings = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.embed_dim,
            padding_idx=0
        )

        # Initialize Performer decoder stack with the given configuration and embeddings
        self.performer_decoder = PerformerDecoder(
            self.config,
            embed_tokens=self.embeddings
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Defines the forward pass of the Performer model.

        Args:
            input_ids (Tensor): Tensor of input token IDs, shaped [batch_size, sequence_length].

        Returns:
            Tensor: Output tensor shaped [batch_size, sequence_length, vocab_size], containing
            the logits over the vocabulary for each token position.
        """
        # Embedding tokens using the embedding layer
        embedding_output = self.embeddings(input_ids)
        
        # Passing embeddings through the Performer decoder stack
        output = self.performer_decoder(embedding_output)
        return output

    def get_params(self) -> dict:
        """
        Extracts and returns model configuration parameters as a dictionary, 
        filtering out non-primitive types for serialization or logging purposes.

        Returns:
            dict: A dictionary of model configuration parameters with primitive type values.
        """
        allowed_types = (int, float, str, bool, Tensor)
        hparams = {key: val for key, val in self.config.to_dict().items() if isinstance(val, allowed_types)}
        return hparams