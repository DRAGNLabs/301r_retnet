import torch
# from train_model import RetNetModel
import transformers
# from torchscale.architecture.config import RetNetConfig

# model = torch.load('scripts/retnet301.pt')
# # print(model.parameters())
# # print("Model's state_dict:")
# # for param_tensor in .load_state_dict(my_state_dict):
# #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

# model = RetNetModel(activation_dropout=0.1, checkpoint_activations=False, dropout=0.1, embed_dim=10, ffn_dim=10, fsdp=True, layers=2, retention_heads=1, max_seq_len=30, value_embed_dim=10, vocab_size=1000)
# model.load_state_dict(torch.load('scripts/retnet301.pt'))
# # model.push_to_hub()  # doesn't work because it's not a transformers model
# # [print(f"{i}: {key}") for i, key in enumerate(model.state_dict().keys())]

# # config = model.RetNetConfig()
# import inspect
# f = inspect.getatt:(model.config)
# # f = inspect.getsource(RetNetConfig)
# # print(f)
# config_dict = {}#model.config.to_dict()
# for attribute in f:
#     config_dict[attribute[0]] = attribute[1]

# # print(config_dict)
# [print(f"{i}: {g}") for i, g in enumerate(f)]
# # print(model.config.decoder_value_embed_dim)
print(f"\n\n\n\hello world\n\n\n")
model = transformers.AutoModelWithLMHead.from_pretrained("dayyass/retnet-1b-random")
print(f"\n\n\n\goodbye world\n\n\n")
# print(config)