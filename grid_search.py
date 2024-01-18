import itertools
import torch
from train_model import train_model

# Hyperparameters ranges to test
learning_rates = [0.01, 0.001]
embed_dims = [768, 1024]
batch_sizes = [16]

# Cartesian product of all hyperparameters
param_combinations = list(itertools.product(learning_rates, embed_dims, batch_sizes))


def evaluate_models(model1, model2, retnet_loss, transformer_loss):
    # Add your code here to evaluate the models on the evaluation data
    return abs(retnet_loss - transformer_loss)


similarity_scores = {}
# Iterate over each combination of parameters
for lr, embed_dim, batch_size in param_combinations:
    print(f"Training with lr={lr}, embed_dim={embed_dim}, batch_size={batch_size}")

    retnet_model, avg_loss_retnet = train_model(embed_dim=embed_dim,
                                                lr=lr,
                                                batch_size=batch_size,
                                                model_type="retnet")

    transformer_model, avg_loss_transformer = train_model(embed_dim=embed_dim,
                                                          lr=lr,
                                                          batch_size=batch_size,
                                                          model_type="transformer")

    # Evaluate the models on the evaluation data
    similarity_score = evaluate_models(retnet_model, transformer_model, avg_loss_retnet, avg_loss_transformer)
    print(f"Similarity score: {similarity_score}")
    similarity_scores[(lr, embed_dim, batch_size)] = similarity_score

# Now to evaluate the best model on the test data
most_similar_params = min(similarity_scores, key=similarity_scores.get)
print(f"========================================\n========================================\n")
print(f"Most similar parameters: {most_similar_params}")
print(f"Similarity score: {similarity_scores[most_similar_params]}")
print(f"========================================\n========================================\n")

most_different_params = max(similarity_scores, key=similarity_scores.get)
print(f"========================================\n========================================\n")
print(f"Most different parameters: {most_different_params}")
print(f"Similarity score: {similarity_scores[most_different_params]}")
print(f"========================================\n========================================\n")
