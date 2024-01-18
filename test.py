import torch
from hugging_face_model import RetNetModel, TransformerModel

retnet = RetNetModel(config='retnet_config/config.json')
retnet.load_state_dict(torch.load('retnet_config/retnet301.pt'))
print("RetNet worked!")
transformer = TransformerModel(config='transformer_config/config.json')
transformer.load_state_dict(torch.load('transformer_config/transformer.pt'))
print("Transformer worked!")


