from train_model import main as train_model
import itertools

param_grid = {
    'embed_dim': [256, 512, 768],
    'ffn_dim': [512, 1024, 2048],
    # Add other parameters here
}

# Create a product of all parameter combinations
param_combinations = list(itertools.product(*param_grid.values()))

for combination in param_combinations:
    params = dict(zip(param_grid.keys(), combination))
    # Call the main function with these parameters
    train_model(**params)