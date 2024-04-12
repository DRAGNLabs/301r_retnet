import torch.nn as nn
from torch import Tensor

from transformers import PreTrainedModel

class HuggingFace(PreTrainedModel):
    """
    This class extends the Hugging Face PreTrainedModel,
    allowing it to be used seamlessly within the Hugging Face ecosystem, including serialization,
    tokenization, and integration with Hugging Face's training and inference pipelines.
    """

    def __init__(self, model, config):
        """ Sets model and config objects
        Args:
            Model (nn.Module): Your model class (ex. Transformer)
            config : A configuration object.
        """
        super().__init__(config)
        self.model = model
        self.config = config

        
    def make_embeddings(self, config):
        """ Create embeddings with index 0 representing padding. """
        text_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.decoder_embed_dim,
            padding_idx=0)
        return text_embeddings


    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Long tensor of dimensions: (batch size, sequence
                length).

        Returns:
            A tensor of dimensions: (batch size, sequence length, vocabulary
                size).
        """
        preds, _ = self.model(x)
        return preds

    def get_params(self) -> dict:
        """ Get model parameters dictionary. """
        allowed_types = (int, float, str, bool, Tensor)
        hparams = self.config.to_dict()
        for key, value in hparams.items():
            if not isinstance(value, allowed_types):
                hparams[key] = str(value)
        return hparams