import copy
import itertools
import sys
import time
import torch
import yaml

from pathlib import Path
from train_model_lightning import train_model
from utils import Struct

def evaluate_models(
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        model1_loss: float,
        model2_loss: float):
    """ Give comparison of two different torch.nn.Module models.
    Args:
        model1 (torch.nn.Module): First instance of model to compare.
        model2 (torch.nn.Module): Second instance of model to compare.
        model1_loss (float): Test loss of first model.
        model2_loss (float): Test loss of second model.

    Returns:
        A float -- positive if model2 is better or negative if otherwise.
    """
    return model1_loss - model2_loss


def grid_search(config: Struct):
    """ Perform grid search on the hyperparameters of the model."""
    # Hyperparameters ranges to test
    learning_rates = [0.01, 0.001, 0.0001]
    embed_dims = [768, 1024, 1280]
    batch_sizes = [16, 32, 64]

    # Cartesian product of all hyperparameters
    param_combinations = list(itertools.product(
        learning_rates,
        embed_dims,
        batch_sizes))

    # Open a CSV file to write the results
    with open(Path(config.root_data_path) / "grid_search_results.csv",
            "w") as results_file:
        # Write header to CSV file
        results_file.write(",".join([
            "Random Seed",
            "Learning Rate",
            "Embedding Dimension",
            "Batch Size",
            "RetNet Avg Loss",
            "Transformer Avg Loss",
            "Similarity Score",
            "RetNet Training Time",
            "Transformer Training Time",
            "Training Time"]) + "\n")

    # Train models with each different permutation of hyperparameters
    similarity_scores = {}
    for lr, embed_dim, batch_size in param_combinations:
        # Prepare seperate config objects to pass
        retnet_config = copy.deepcopy(config)
        retnet_config.lr = lr
        retnet_config.embed_dim = embed_dim
        retnet_config.batch_size = batch_size
        retnet_config.model_type = "retnet"

        transformer_config = copy.deepcopy(config)
        transformer_config.lr = lr
        transformer_config.embed_dim = embed_dim
        transformer_config.batch_size = batch_size
        transformer_config.model_type = "transformer"

        start_time = time.time()

        # Train RetNet model
        retnet_start_time = time.time()
        retnet_model, avg_loss_retnet = train_model(retnet_config)
        retnet_training_time=time.time() - retnet_start_time

        # Train Transformer model with same hyperparameters as RetNet model
        transformer_start_time = time.time()
        transformer_model, avg_loss_transformer = train_model(transformer_config)
        transformer_training_time = time.time() - transformer_start_time

        # Track how much time both models combined took to train
        total_time = time.time() - start_time

        # Compare both models
        similarity_score = evaluate_models(
            model1=retnet_model,
            model2=transformer_model,
            model1_loss=avg_loss_retnet,
            model2_loss=avg_loss_transformer)

        # Record results in CSV
        with open(Path(config.root_data_path) / "grid_search_results.csv",
                "a") as results_file:
            results_file.write(",".join(map(str, [
                config.rand_seed,
                lr,
                embed_dim,
                batch_size,
                avg_loss_retnet,
                avg_loss_transformer,
                similarity_score,
                retnet_training_time,
                transformer_training_time,
                total_time])) + "\n")

    # Store similarity score for comparison later
    similarity_scores[(lr, embed_dim, batch_size)] = similarity_score

    # Find the most and least similar parameters, just for fun
    most_similar_params = min(similarity_scores, key=similarity_scores.get)
    most_diff_params = max(similarity_scores, key=similarity_scores.get)

    # Save comparison of similarity results
    with open(Path(config.root_data_path) / "grid_search_results.csv",
            "a") as results_file:
        results_file.write(",".join(map(str, [
            "Most Similar Parameters",
            *most_similar_params,
            similarity_scores[most_similar_params]])) + "\n")
        results_file.write(",".join(map(str, [
            "Most Different Parameters",
            *most_diff_params,
            similarity_scores[most_diff_params]])) + "\n")


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    grid_search(config)
