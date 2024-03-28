from models.hugging_face.transformer import TransformerHF
from models.lightning.lightning import LightningModel
from torchscale.architecture.config import DecoderConfig


class TransformerLightning(LightningModel):
    def __init__(self, config):
        hf_config = DecoderConfig(
            decoder_embed_dim=config.embed_dim,
            decoder_attention_heads=config.heads,
            decoder_ffn_embed_dim=config.ffn_dim,
            decoder_layers=config.layers,
            dropout=config.dropout,
            activation_dropout=config.activation_dropout,
            vocab_size=config.vocab_size)
        super().__init__(config, hf_config, TransformerHF)