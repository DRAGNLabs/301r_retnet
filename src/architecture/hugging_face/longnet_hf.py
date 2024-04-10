from torchscale.architecture.decoder import Decoder
from architecture.hugging_face.hugging_face import HuggingFace

class LongNetHF(HuggingFace):
    def __init__(self, config):
        model = Decoder(config, embed_tokens=self.make_embeddings(config))
        super().__init__(model, config)