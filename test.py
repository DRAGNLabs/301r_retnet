from torchscale.architecture.config import RetNetConfig
from hugging_faceify import hugging_faceify

config = RetNetConfig()
config.save_pretrained('ret_config')
ret = hugging_faceify('scripts/retnet301.pt', 'ret_config/config.json')


