import itertools
import torch
import csv
import time
from train_model import train_model
from argparse import ArgumentParser

def grid_search(dataset_dir, dataset_name, dataset_subset, data_dir, dataset_feature):
    """ Perform grid search on the hyperparameters of the model.

    Args:
        dataset_dir (str): Path to directory in which Hugging Face datasets are
            downloaded.
        dataset_name (str): Name of Hugging Face dataset.
        dataset_subset (str): Configuration/subset of dataset to use.
    """

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
            retnet_model, avg_loss_retnet = train_model(embed_dim=embed_dim, lr=lr, batch_size=batch_size, 
                                                        model_type="retnet", dataset_dir=dataset_dir, 
                                                        dataset_name=dataset_name, dataset_subset=dataset_subset,
                                                        data_dir=data_dir, dataset_feature=dataset_feature)
            retnet_training_time = time.time() - retnet_start_time()
            
            transformer_start_time = time.time()
            transformer_model, avg_loss_transformer = train_model(embed_dim=embed_dim, lr=lr, batch_size=batch_size, 
                                                                  model_type="transformer", dataset_dir=dataset_dir, 
                                                                  dataset_name=dataset_name, dataset_subset=dataset_subset,
                                                                  data_dir=data_dir, dataset_feature=dataset_feature)
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


if __name__ == "__main__":
    # Initialize, setup, and parse the argument parser
    parser = ArgumentParser(prog="Grid Search")

    parser.add_argument("--dataset-dir", type=str, required=True,
        help="Path to directory in which Hugging Face datasets are downloaded.")
    parser.add_argument("--dataset-name", type=str, required=True,
        help="Hugging Face dataset name. Should also set --dataset-subset.")
    parser.add_argument("--dataset-subset", type=str, required=True,
        help="Subset/config to use for Hugging Face dataset.")
    parser.add_argument("--data-dir", type=str, required=True,
        help="No clue what this is for.")
    parser.add_argument("--dataset-feature", type=str, required=True,
        help="No clue what this is for.")

    args = parser.parse_args()

    grid_search(args.dataset_dir, args.dataset_name, args.dataset_subset, args.data_dir, args.dataset_feature)
