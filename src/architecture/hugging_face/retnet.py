from torchscale.architecture.retnet import RetNetDecoder
from architecture.hugging_face.hugging_face import HuggingFace

class RetNetHF(HuggingFace):
    def __init__(self, config):
        model = RetNetDecoder(config, embed_tokens=self.make_embeddings(config))
        super().__init__(model, config)