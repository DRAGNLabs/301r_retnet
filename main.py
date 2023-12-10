
print("import torch")
import torch
print("import RetNetConfig")
from torchscale.architecture.config import RetNetConfig
print("import RetNetDecoder")
from torchscale.architecture.retnet import RetNetDecoder

print("Config of config")
config = RetNetConfig(vocab_size=64000)
print("Make RetNetDecoder.")
retnet = RetNetDecoder(config)
print(retnet)
print("Success. Done.")