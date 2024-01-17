from torch import nn, Tensor
from transformers import PreTrainedModel
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder

class InferenceModel(PreTrainedModel):
    def __init__(
            self,
            # embed_dim: int,
            # value_embed_dim: int,
            # retention_heads: int,
            # ffn_dim: int,
            # layers: int,
            # dropout: float,
            # activation_dropout: float,
            # vocab_size: int,    
            # fsdp: bool,
            # max_seq_len: int,
            config_path:str=None):
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
        assert type(config_path) is str or config_path is None, "config_path must be a string or None;\nto pass in a config object, first save with config.save_pretrained('your_path'),\nthen pass in 'your_path'"
        self.config = RetNetConfig.from_pretrained(config_path) if type(config_path)==str else RetNetConfig()
        super().__init__(self.config)

        # Store hyperparameters
        self.model_params = {
            "embed_dim": self.config.decoder_embed_dim,
            "value_embed_dim": self.config.decoder_value_embed_dim,
            "retention_heads": self.config.decoder_retention_heads,
            "ffn_dim": self.config.decoder_ffn_embed_dim,
            "layers": self.config.decoder_layers,
            "dropout": self.config.dropout,
            "activation_dropout": self.config.activation_dropout,
            "vocab_size": self.config.vocab_size,
            "fsdp": self.config.fsdp,
            "max_seq_len": self.config.max_seq_len
        }

        # Create embeddings with index 0 representing padding
        self.config.vocab_size = 50000
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