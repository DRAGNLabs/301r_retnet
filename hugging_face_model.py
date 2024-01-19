import torch.nn as nn

from torch import Tensor
from typing import Optional, Union
from torchscale.architecture.config import DecoderConfig, RetNetConfig
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.retnet import RetNetDecoder
from transformers import PreTrainedModel, AutoModel, AutoConfig

class RetNetModel(PreTrainedModel):
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

        super().__init__(config)

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
        return self.model_params


class TransformerModel(PreTrainedModel):
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
        return self.model_params