import torch.nn as nn

from torch import Tensor
from torchscale.architecture.config import DecoderConfig, RetNetConfig
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.retnet import RetNetDecoder
from transformers import PreTrainedModel

class RetNetModel(PreTrainedModel):
    """ Create model with RetNet architecture. """
    def __init__(
            self,
            embed_dim: int=768,
            value_embed_dim: int=1280,
            retention_heads: int=3,
            ffn_dim: int=1280,
            layers: int=12,
            dropout: float=0.1,
            activation_dropout: float=0.0,
            vocab_size: int=50265,
            fsdp: bool=False,
            max_seq_len: int=512,
            config: str=None):
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
        if config:
            self.config = RetNetConfig.from_pretrained(config)
        else:
            self.config = RetNetConfig(
                decoder_embed_dim=embed_dim,
                decoder_value_embed_dim=value_embed_dim,
                decoder_retention_heads=retention_heads,
                decoder_ffn_embed_dim=ffn_dim,
                decoder_layers=layers,
                dropout=dropout,
                activation_dropout=activation_dropout,
                vocab_size=vocab_size,
                fsdp=fsdp)

        super().__init__(self.config)

        # Create embeddings with index 0 representing padding
        text_embeddings = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.decoder_embed_dim,
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
    def __init__(
            self,
            embed_dim: int = 768,
            value_embed_dim: int = 1280,
            attention_heads: int = 3,
            ffn_dim: int = 1280,
            layers: int = 12,
            dropout: float = 0.1,
            activation_dropout: float = 0.0,
            vocab_size: int = 50265,
            fsdp: bool = False,
            max_seq_len: int = 512,
            config: str = None):
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
        
        if config:
            self.config = DecoderConfig.from_pretrained(config)
        else:
            self.config = DecoderConfig(
                decoder_embed_dim=embed_dim,
                decoder_value_embed_dim=value_embed_dim,
                decoder_attention_heads=attention_heads,
                decoder_ffn_embed_dim=ffn_dim,
                decoder_layers=layers,
                dropout=dropout,
                activation_dropout=activation_dropout,
                vocab_size=vocab_size,
                fsdp=fsdp)
        
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