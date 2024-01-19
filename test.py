from transformers import AutoModel, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from hugging_face_model import RetNetModel, TransformerModel
from torchscale.architecture.config import RetNetConfig, DecoderConfig


AutoConfig.register('retnet', RetNetConfig)
AutoModel.register(RetNetConfig, RetNetModel)
AutoModelForCausalLM.register(RetNetConfig, RetNetModel)
AutoTokenizer.register(RetNetConfig, RetNetModel)

test = AutoModelForCausalLM.from_pretrained('retnet')

tokenizer = AutoTokenizer.from_pretrained('weights/2024-01-18-16:06:39_retnet_28452480')
breakpoint()


