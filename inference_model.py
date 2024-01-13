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
            config_path=None):
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
        config = RetNetConfig.from_pretrained(config_path) if config_path else RetNetConfig()
        super().__init__(config)

        # Store hyperparameters
        self.model_params = {
            "embed_dim": config.decoder_embed_dim,
            "value_embed_dim": config.decoder_value_embed_dim,
            "retention_heads": config.decoder_retention_heads,
            "ffn_dim": config.decoder_ffn_embed_dim,
            "layers": config.decoder_layers,
            "dropout": config.dropout,
            "activation_dropout": config.activation_dropout,
            "vocab_size": config.vocab_size,
            "fsdp": config.fsdp,
            "max_seq_len": config.max_seq_len
        }

        # Create embeddings with index 0 representing padding
        text_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.decoder_embed_dim,
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