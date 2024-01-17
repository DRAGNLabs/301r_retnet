import torch

from inference_model import InferenceModel


def hugging_faceify(path_to_model:str, path_to_config:str=None):
    """
    path_to_model: path to the pytorch model
    path_to_config: path to the config file
    """
    retnet = InferenceModel(path_to_config)
    loaded = torch.load(path_to_model)
    print(type(loaded), len(loaded.keys()))
    retnet.decoder_stack.load_state_dict(loaded)
   
    print('Done!')
    return retnet

# hugging_faceify('scripts/retnet301.pt', 'test_config/config.json')