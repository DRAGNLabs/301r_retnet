from transformers import PretrainedConfig
from typing import List  # just for type hinting


class RetnetConfig(PretrainedConfig):
    # model_type = "retnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)

        {
  "activation_dropout": 0.1,
#   "activation_fn": "swish",
  "architectures": ["RetNetForCausalLM"],
  "decoder_embed_dim": 10,
  "decoder_ffn_embed_dim": 10,
  "decoder_layers": 2,
#   "decoder_normalize_before": true,
  "decoder_retention_heads": 1,
  "decoder_value_embed_dim": 10,
#   "deepnorm": false,
#   "drop_path_rate": 0.0,
  "dropout": 0.1,
#   "eos_token_id": 11,
#   "forward_impl": "parallel",
#   "initializer_range": 0.02,
#   "is_decoder": true,
#   "layernorm_embedding": true,
#   "layernorm_eps": 1e-06,
  "model_type": "retnet",
#   "no_scale_embedding": true,
#   "output_retentions": false,
#   "pad_token_id": 11,
  "recurrent_chunk_size": 30,
#   "subln": true,
#   "tie_word_embeddings": false,
#   "torch_dtype": "float32",
#   "transformers_version": "4.35.0",
#   "unk_token_id": 11,
#   "use_cache": true,
#   "use_ffn_rms_norm": false,
#   "use_glu": true,
#   "use_lm_decay": false,
  "vocab_size": 1000
#   "z_loss_coeff": 0.0
}
        
#TODO: fdsp?, checkpoint-activations? These don't seem to have parallel arguments in this file.