from torchscale.architecture.decoder import Decoder
from models.hugging_face.hugging_face import HuggingFace

class TransformerHF(HuggingFace):
    def __init__(self, config):
        self.config = config
        model = Decoder(config, embed_tokens=self.make_embeddings(config))
        super().__init__(model, config)