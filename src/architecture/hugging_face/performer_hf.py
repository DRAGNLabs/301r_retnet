from torchscale.architecture.performer import PerformerDecoder
from architecture.hugging_face.hugging_face import HuggingFace

class PerformerHF(HuggingFace):
    def __init__(self, config):
        model = PerformerDecoder(
            config,
            embed_tokens=self.make_embeddings(config),
            max_seq_len=config.max_seq_len,
            num_tokens=config.vocab_size,
            dim_head=config.dim_head,
            dim=config.embed_dim,
        )
        super().__init__(model, config)