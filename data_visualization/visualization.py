# This creates all the visualizations for the paper from the included files in the local datarun folder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the data
retnet_df = pd.read_csv('datarun/retnet_grid_search_results.csv')
transformer_df = pd.read_csv('datarun/transformer_grid_search_results.csv')

# Plot the results
x = 'Learning Rate'
y = 'Embedding Dimension'

def save_figure(x, y):
    plt.figure(figsize=(10, 7))
    plt.scatter(retnet_df[x], retnet_df[y], label='RetNet')
    # plt.scatter(transformer_df[x], transformer_df[y], label='Transformer')
    plt.legend()
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} vs. {y}')

    # Make the figures directory if it doesn't exist
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/{x}_vs_{y}.png')

save_figure(x, y)

