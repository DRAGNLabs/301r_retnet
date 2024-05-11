import copy
import itertools
import sys
import time
import yaml

from pathlib import Path
from train_model import train_model
from utils import Struct

def evaluate_models(
        model1_path: str,
        model2_path: str,
        model1_loss: float,
        model2_loss: float):
    """ Give comparison of two different models.
    Args:
        model1_path (str): Path to first instance of model to compare.
        model2_path (str): Path to second instance of model to compare.
        model1_loss (float): Test loss of first model.
        model2_loss (float): Test loss of second model.

    Returns:
        A float -- positive if model2 is better or negative if otherwise.
    """
    return model1_loss - model2_loss


def grid_search(config: Struct):
    """ Perform grid search on the hyperparameters of the model.
    Args:
        config (Struct): A Struct object with all configuration fields.
    """
        
    assert not isinstance(config.learning_rate, str), f"Error: 'config.learning_rate' \
should not be a string (you put {config.learning_rate}).\nNote: You cannot use 'find' \
feature in conjunction with grid search."

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
            "LongNet Avg Loss",
            "RetNet Avg Loss",
            "Transformer Avg Loss",
            "Similarity Score Between RetNet and Transformer",
            "Similarity Score Between RetNet and LongNet",
            "Similarity Score Between LongNet and Transformer",
            "LongNet Training Time",
            "RetNet Training Time",
            "Transformer Training Time",
            "Training Time"]) + "\n")

    # Train models with each different permutation of hyperparameters
    similarity_retnet_transformer = {}
    similarity_retnet_longnet = {}
    similarity_longnet_transformer = {}
    for lr, embed_dim, batch_size in param_combinations:
        # Prepare seperate config objects to pass
        longnet_config = copy.deepcopy(config)
        longnet_config.lr = lr
        longnet_config.embed_dim = embed_dim
        longnet_config.batch_size = batch_size
        longnet_config.model_type = "longnet"
        
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


        # Train models
        longnet_start_time = time.time()
        longnet_model_path, longnet_best_score = train_model(longnet_config)
        longnet_training_time=time.time() - longnet_start_time

        retnet_start_time = time.time()
        retnet_model_path, retnet_best_score = train_model(retnet_config)
        retnet_training_time=time.time() - retnet_start_time

        transformer_start_time = time.time()
        transformer_model_path, transformer_best_score = train_model(transformer_config)
        transformer_training_time = time.time() - transformer_start_time

        # Track how much time both models combined took to train
        total_time = time.time() - start_time

        # Compare both models
        similarity_score_1 = evaluate_models(
            model1_path=retnet_model_path,
            model2_path=transformer_model_path,
            model1_loss=retnet_best_score,
            model2_loss=transformer_best_score)
        similarity_score_2 = evaluate_models(
            model1_path=retnet_model_path,
            model2_path=longnet_model_path,
            model1_loss=retnet_best_score,
            model2_loss=longnet_best_score)
        similarity_score_3 = evaluate_models(
            model1_path=longnet_model_path,
            model2_path=transformer_model_path,
            model1_loss=longnet_best_score,
            model2_loss=transformer_best_score)

        # Record results in CSV
        with open(Path(config.root_data_path) / "grid_search_results.csv",
                "a") as results_file:
            results_file.write(",".join(map(str, [
                config.rand_seed,
                lr,
                embed_dim,
                batch_size,
                longnet_best_score,
                retnet_best_score,
                transformer_best_score,
                similarity_score_1,
                similarity_score_2,
                similarity_score_3,
                longnet_training_time,
                retnet_training_time,
                transformer_training_time,
                total_time])) + "\n")

    # Store similarity score for comparison later
    similarity_retnet_transformer[(lr, embed_dim, batch_size)] = similarity_score_1
    similarity_retnet_longnet[(lr, embed_dim, batch_size)] = similarity_score_2
    similarity_longnet_transformer[(lr, embed_dim, batch_size)] = similarity_score_3

    # Find the most and least similar parameters, just for fun
    most_similar_params_1 = min(similarity_retnet_transformer, key=similarity_retnet_transformer.get)
    most_diff_params_1 = max(similarity_retnet_transformer, key=similarity_retnet_transformer.get)
    most_similar_params_2 = min(similarity_retnet_longnet, key=similarity_retnet_longnet.get)
    most_diff_params_2 = max(similarity_retnet_longnet, key=similarity_retnet_longnet.get)
    most_similar_params_3 = min(similarity_longnet_transformer, key=similarity_longnet_transformer.get)
    most_diff_params_3 = max(similarity_longnet_transformer, key=similarity_longnet_transformer.get)

    # Save comparison of similarity results
    with open(Path(config.root_data_path) / "grid_search_results.csv",
            "a") as results_file:
        results_file.write(",".join(map(str, [
            "Most Similar Parameters between RetNet and Transformer",
            *most_similar_params_1,
            similarity_retnet_transformer[most_similar_params_1]])) + "\n")
        results_file.write(",".join(map(str, [
            "Most Different Parameters between RetNet and Transformer",
            *most_diff_params_1,
            similarity_retnet_transformer[most_diff_params_1]])) + "\n")
        results_file.write(",".join(map(str, [
            "Most Similar Parameters between RetNet and LongNet",
            *most_similar_params_2,
            similarity_retnet_longnet[most_similar_params_2]])) + "\n")
        results_file.write(",".join(map(str, [
            "Most Different Parameters between RetNet and LongNet",
            *most_diff_params_2,
            similarity_retnet_longnet[most_diff_params_2]])) + "\n")
        results_file.write(",".join(map(str, [
            "Most Similar Parameters between LongNet and Transformer",
            *most_similar_params_3,
            similarity_longnet_transformer[most_similar_params_3]])) + "\n")
        results_file.write(",".join(map(str, [
            "Most Different Parameters between LongNet and Transformer",
            *most_diff_params_3,
            similarity_longnet_transformer[most_diff_params_3]])) + "\n")


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)
        
    grid_search(config)
