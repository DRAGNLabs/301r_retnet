import csv
import itertools
import time
import torch

from argparse import ArgumentParser
from pathlib import Path
from train_model_lightning import train_model

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


def grid_search(
        data_dir: str,
        datasets_dir: str,
        dataset_feature: str,
        dataset_name: str,
        dataset_subset: str,
        tokenizer_folder: str,
        num_devices: str,
        train_data: str,
        validation_data: str,
        test_data: str):
    """ Perform grid search on the hyperparameters of the model.

    Args:
        data_dir (str): Path to directory where all data except datasets are
            saved.
        datasets_dir (str): Path to directory in which Hugging Face datasets are
            downloaded.
        dataset_feature (str): Hugging Face dataset feature/column to use.
        dataset_name (str): Hugging Face dataset name. Should also set
            '--dataset-subset'.
        dataset_subset (str): Subset/config to use for Hugging Face dataset.
    """
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
    with open(Path(data_dir) / "grid_search_results.csv", "w") as results_file:
        # Write header to CSV file
        results_file.write(",".join([
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
        start_time = time.time()

        # Train RetNet model
        retnet_start_time = time.time()
        retnet_model, avg_loss_retnet = train_model(
            embed_dim=embed_dim,
            lr=lr,
            batch_size=batch_size,
            model_type="retnet",
            datasets_dir=datasets_dir,
            dataset_name=dataset_name,
            dataset_subset=dataset_subset,
            data_dir=data_dir,
            dataset_feature=dataset_feature,
            tboard_dir="/tmp/tboard_logs",
            num_devices=num_devices,
            tokenizer_folder=tokenizer_folder,
            train_data = train_data,
            validation_data = validation_data,
            test_data = test_data)
        retnet_training_time = time.time() - retnet_start_time

        # Train Transformer model with same hyperparameters as RetNet model
        transformer_start_time = time.time()
        transformer_model, avg_loss_transformer = train_model(
            embed_dim=embed_dim,
            lr=lr,
            batch_size=batch_size,
            model_type="transformer",
            datasets_dir=datasets_dir,
            dataset_name=dataset_name,
            dataset_subset=dataset_subset,
            data_dir=data_dir,
            dataset_feature=dataset_feature,
            tboard_dir="/tmp/tboard_logs",
            num_devices=num_devices,
            tokenizer_folder=tokenizer_folder,
            train_data = train_data,
            validation_data = validation_data,
            test_data = test_data)
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
        with open(Path(data_dir) / "grid_search_results.csv",
                "a") as results_file:
            results_file.write(",".join(map(str, [
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
    with open(Path(data_dir) / "grid_search_results.csv", "a") as results_file:
        results_file.write(",".join(map(str, [
            "Most Similar Parameters",
            *most_similar_params,
            similarity_scores[most_similar_params]])) + "\n")
        results_file.write(",".join(map(str, [
            "Most Different Parameters",
            *most_diff_params,
            similarity_scores[most_diff_params]])) + "\n")


if __name__ == "__main__":
    # Initialize, setup, and parse the argument parser
    parser = ArgumentParser(prog="Grid Search")

    parser.add_argument("--data-dir", type=str, required=True,
        help="Path to directory where all data except datasets are saved.")
    parser.add_argument("--dataset-feature", type=str, default="text",
        help="Hugging Face dataset feature/column to use.")
    parser.add_argument("--dataset-name", type=str, default="wikitext",
        help="Hugging Face dataset name. Should also set --dataset-subset.")
    parser.add_argument("--dataset-subset", type=str, default="wikitext-2-v1",
        help="Subset/config to use for Hugging Face dataset.")
    parser.add_argument("--datasets-dir", type=str, required=True,
        help="Path to directory in which Hugging Face datasets are downloaded.")
    parser.add_argument("--tokenizer-folder", type= str, required=True,
        help="Path to the file where the tokenizer will be saved")
    parser.add_argument("--num-devices", type= str, required=True,
        help="Number of gpus to train on")
    parser.add_argument("--train-data", type= str, required=True,
        help="Direct path to tokenized train data")
    parser.add_argument("--validation-data", type= str, required=True,
        help="Direct path to tokenized validation data")
    parser.add_argument("--test-data", type= str, required=True,
        help="Direct path to tokenized test data")
    
    args = parser.parse_args()
    grid_search(**vars(args))
