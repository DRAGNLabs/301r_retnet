import csv
import itertools
import time
import torch

from argparse import ArgumentParser
from pathlib import Path
from train_model import train_model

def evaluate_models(model1, model2, model1_loss, model2_loss):
    return abs(model1_loss - model2_loss)


def grid_search(data_dir, dataset_dir, dataset_feature, dataset_name, dataset_subset):
    """ Perform grid search on the hyperparameters of the model.

    Args:
        data_dir (str): Path to directory where all data except datasets are saved.
        dataset_dir (str): Path to directory in which Hugging Face datasets are downloaded.
        dataset_feature (str): Hugging Face dataset feature/column to use.
        dataset_name (str): Hugging Face dataset name. Should also set --dataset-subset.
        dataset_subset (str): Subset/config to use for Hugging Face dataset.
    """

    # Hyperparameters ranges to test
    learning_rates = [0.01, 0.001, 0.0001]
    embed_dims = [768, 1024, 1282]
    batch_sizes = [16, 32, 64]

    # Cartesian product of all hyperparameters
    param_combinations = list(itertools.product(learning_rates, embed_dims, batch_sizes))

    # Open a CSV file to write the results
    with open(Path(data_dir) / 'grid_search_results.csv', 'w') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(['Learning Rate', 'Embedding Dimension', 'Batch Size',
                        'RetNet Avg Loss', 'Transformer Avg Loss', 'Similarity Score',
                        'RetNet Training Time', 'Transformer Training Time', 'Training Time'])

        similarity_scores = {}
        for lr, embed_dim, batch_size in param_combinations:
            start_time = time.time()

            retnet_start_time = time.time()
            retnet_model, avg_loss_retnet = train_model(embed_dim=embed_dim, lr=lr, batch_size=batch_size,
                                                        model_type="retnet", dataset_dir=dataset_dir,
                                                        dataset_name=dataset_name, dataset_subset=dataset_subset,
                                                        data_dir=data_dir, dataset_feature=dataset_feature,
                                                        tboard_dir="/tmp/tboard_logs")
            retnet_training_time = time.time() - retnet_start_time

            transformer_start_time = time.time()
            transformer_model, avg_loss_transformer = train_model(embed_dim=embed_dim, lr=lr, batch_size=batch_size,
                                                                  model_type="transformer", dataset_dir=dataset_dir,
                                                                  dataset_name=dataset_name, dataset_subset=dataset_subset,
                                                                  data_dir=data_dir, dataset_feature=dataset_feature,
                                                                  tboard_dir="/tmp/tboard_logs")
            transformer_training_time = time.time() - transformer_start_time

            total_time = time.time() - start_time
            similarity_score = evaluate_models(retnet_model, transformer_model, avg_loss_retnet, avg_loss_transformer)

            writer.writerow([lr, embed_dim, batch_size,
                            avg_loss_retnet, avg_loss_transformer, similarity_score,
                            retnet_training_time, transformer_training_time, total_time])
            similarity_scores[(lr, embed_dim, batch_size)] = similarity_score

        # Find the most and least similar parameters, just for fun
        most_similar_params = min(similarity_scores, key=similarity_scores.get)
        most_different_params = max(similarity_scores, key=similarity_scores.get)

        writer.writerow(['Most Similar Parameters', *most_similar_params, similarity_scores[most_similar_params]])
        writer.writerow(['Most Different Parameters', *most_different_params, similarity_scores[most_different_params]])


if __name__ == "__main__":
    # Initialize, setup, and parse the argument parser
    parser = ArgumentParser(prog="Grid Search")

    parser.add_argument("--data-dir", type=str, required=True,
        help="Path to directory where all data except datasets are saved.")
    parser.add_argument("--dataset-dir", type=str, required=True,
        help="Path to directory in which Hugging Face datasets are downloaded.")
    parser.add_argument("--dataset-feature", type=str, default="text",
        help="Hugging Face dataset feature/column to use.")
    parser.add_argument("--dataset-name", type=str, default="wikitext",
        help="Hugging Face dataset name. Should also set --dataset-subset.")
    parser.add_argument("--dataset-subset", type=str, default="wikitext-2-v1",
        help="Subset/config to use for Hugging Face dataset.")

    args = parser.parse_args()
    grid_search(**vars(args))
