from architecture.hugging_face.LongNet_hf import LongNetHF
from architecture.lightning.lightning import LightningModel
from torchscale.architecture.config import LongNetConfig


class LongNetLightning(LightningModel):
    def __init__(self, config):
        hf_config = LongNetConfig(
            decoder_embed_dim=config.embed_dim,
            decoder_value_embed_dim=config.value_embed_dim,
            decoder_attention_heads=config.heads,
            decoder_ffn_embed_dim=config.ffn_dim,
            decoder_layers=config.layers,
            dropout=config.dropout,
            activation_dropout=config.activation_dropout,
            vocab_size=config.vocab_size,
            max_seq_len=config.seq_len,
            dim_head=config.dim_head,  # Dimension of each attention head
            local_heads=config.local_heads,  # Number of local attention heads
            local_window_size=config.local_window_size,  # Size of the local attention window
            nb_features=config.nb_features,  # Number of random features for kernel approximation, use 'null' or None here
            generalized_attention=config.generalized_attention,  # Use generalized attention mechanism
            kernel_fn=config.kernel_fn,  # Kernel function used in the generalized attention mechanism
            no_projection=config.no_projection,  # Disable linear projection in attention
            qkv_bias=config.qkv_bias,  # Use bias in QKV projection
            attn_out_bias=config.attn_out_bias,  # Use bias in attention output projection
        )
        super().__init__(config, hf_config, LongNetHF)