import itertools
import torch
import csv
import time
from train_model import train_model

# Hyperparameters ranges to test
learning_rates = [0.01, 0.001, 0.0001]
embed_dims = [768, 1023, 1281]
batch_sizes = [16, 32, 64]

# Cartesian product of all hyperparameters
param_combinations = list(itertools.product(learning_rates, embed_dims, batch_sizes))

def evaluate_models(model1, model2, model1_loss, model2_loss):
    return abs(model1_loss - model2_loss)


# Open a CSV file to write the results
with open('model_training_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Learning Rate', 'Embedding Dimension', 'Batch Size', 
                     'RetNet Avg Loss', 'Transformer Avg Loss', 'Similarity Score', 
                     'RetNet Training Time', 'Transformer Training Time', 'Training Time'])

    similarity_scores = {}
    counter = 0
    for lr, embed_dim, batch_size in param_combinations:
        start_time = time.time()
        
        retnet_start_time = time.time()
        retnet_model, avg_loss_retnet = train_model(embed_dim=embed_dim, lr=lr, batch_size=batch_size, model_type="retnet")
        retnet_training_time = time.time() - retnet_start_time()
        
        transformer_start_time = time.time()
        transformer_model, avg_loss_transformer = train_model(embed_dim=embed_dim, lr=lr, batch_size=batch_size, model_type="transformer")
        transformer_training_time = time.time() - transformer_start_time()
        
        training_time = time.time() - start_time
        similarity_score = evaluate_models(retnet_model, transformer_model, avg_loss_retnet, avg_loss_transformer)

        writer.writerow([lr, embed_dim, batch_size, 
                         avg_loss_retnet, avg_loss_transformer, similarity_score, 
                         retnet_training_time, transformer_training_time, training_time])
        similarity_scores[(lr, embed_dim, batch_size)] = similarity_score

    # Find the most and least similar parameters, just for fun
    most_similar_params = min(similarity_scores, key=similarity_scores.get)
    most_different_params = max(similarity_scores, key=similarity_scores.get)

    writer.writerow(['Most Similar Parameters', *most_similar_params, similarity_scores[most_similar_params]])
    writer.writerow(['Most Different Parameters', *most_different_params, similarity_scores[most_different_params]])
