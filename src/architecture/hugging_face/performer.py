from torchscale.architecture.performer import PerformerDecoder
from models.hugging_face.hugging_face import HuggingFace

class PerformerHF(HuggingFace):
    def __init__(self, config):
        model = PerformerDecoder(
            self.config,
            embed_tokens=self.get_embeddings(config),
            max_seq_len=self.config.max_seq_len,
            num_tokens=self.config.vocab_size,
            dim_head=self.config.dim_head,
            dim=self.config.embed_dim,
        )
        super().__init__(model, config)