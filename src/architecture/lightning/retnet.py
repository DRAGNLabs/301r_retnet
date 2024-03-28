from architecture.hugging_face.retnet import RetNetHF
from architecture.lightning.lightning import LightningModel
from torchscale.architecture.config import RetNetConfig


class RetNetLightning(LightningModel):
    def __init__(self, config):
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
        super().__init__(config, hf_config, RetNetHF)