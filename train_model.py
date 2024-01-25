import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from datasets import (load_dataset as load_ds)
from datetime import datetime
from hugging_face_model import RetNetModelHF, TransformerModelHF
from pathlib import Path
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary as model_summary
from torchscale.architecture.config import RetNetConfig, DecoderConfig
from tqdm import tqdm
from transformers import set_seed, AutoConfig, AutoModel, AutoModelForCausalLM, PreTrainedTokenizerFast

# Allow torch to run float32 matrix multiplications in lower precision for
# better performance while training if hardware is capable
torch.backends.cuda.matmul.allow_tf32 = True

def train_model(
        activation_dropout: float=0.0,
        batch_size: int=8,
        checkpoints: bool=False,
        data_dir: str="/tmp/data",
        datasets_dir: str="/tmp/data/datasets",
        dataset_name: str="wikitext",
        device: str="cuda",
        dropout: float=0.1,
        embed_dim: int=80,
        epochs: int=1,
        ffn_dim: int=12,
        fsdp: bool=False,
        heads: int=4,
        layers: int=2,
        lr: float=0.001,
        model_type: str="retnet",
        rand_seed: bool=None,
        seq_len: int=128,
        tboard_dir: str="/tmp/tboard_logs",
        tokenizer_folder: str=None,
        val_freq: int=3,
        value_embed_dim: int=12,
        vocab_size: int=32768):
    """ Use parameters to run train_model().
        Args:
            activation_dropout (float): Probability of an element to be zeroed
                during dropout after activation between FFN layers.
            batch_size (int): Batch size.
            checkpoints (bool): Save model checkpoints while training.
            data_dir (str): Path to directory where all data except datasets are
                saved.
            datasets_dir (str): Path to directory in which Hugging Face datasets
                are downloaded.
            dataset_name (str): Hugging Face dataset name.
            device (str): Device to use (ex: 'cpu', 'cuda', or 'cuda:0').
            dropout (float): Probability of an element to be zeroed during
                dropout.
            embed_dim (int): Embedding dimension size of each token.
            epochs (int): Number of epochs to train for.
            ffn_dim (int): Hidden layer size of Feed Forward Network (FFN).
            fsdp (bool): Whether to shard Module parameters across data parallel
                workers or not (with the FairScale library).
            heads (int): Number of heads. Head architecture changes based on
                model.
            layers (int): Number of retention network layers.
            lr (float): Learning rate of model to train.
            model_type (str): Name of model architecture to train.
            rand_seed (int): Random seed to use, allowing more reproducible
                results.
            seq_len (int): Sequence length (context window size).
            tboard_dir (str): Path to directory to save TensorBoard logs in.
            val_freq (int): Number of times to run validation per epoch during
                training.
            value_embed_dim (int): Value embed dimension size.
            vocab_size (int): Maximum vocabulary size (number of unique tokens
                in vocabulary.

        Returns:
            A tuple of the trained model instance and the test average loss.
    """
    # Store all the parameters, which are the only locals at this point, as dict
    arg_dict = locals()

    # Test that the head dimension will be an even, whole number
    assert embed_dim % (heads * 2) == 0, \
        "Head Dimension must be even to perform Rotary Position Embedding " + \
        f"({embed_dim} / {heads} = {embed_dim / heads} " + \
        "-- not an even, whole number)! Try changing the Embedding " + \
        "Dimension or number of heads."

    # Test that the value embedding dimension is divisible by number of heads
    assert value_embed_dim % heads == 0, \
        "Value Embed Dimension not divisible by number of heads " + \
        f"({value_embed_dim} % {heads} != 0)!"

    # Set random seeds for torch, numpy, random, etc. with transformers library
    if rand_seed is not None:
        set_seed(rand_seed)

    # Create requested model
    if model_type == "retnet":
        AutoConfig.register("retnet", RetNetConfig)
        AutoModel.register(RetNetConfig, RetNetModelHF)
        AutoModelForCausalLM.register(RetNetConfig, RetNetModelHF)
        config = RetNetConfig(
            decoder_embed_dim=embed_dim,
            decoder_value_embed_dim=value_embed_dim,
            decoder_retention_heads=heads,
            decoder_ffn_embed_dim=ffn_dim,
            decoder_layers=layers,
            dropout=dropout,
            activation_dropout=activation_dropout,
            vocab_size=vocab_size,
            fsdp=fsdp,
            max_seq_len=seq_len)
        model = RetNetModelHF(config)
        
    elif model_type == "transformer":
        AutoConfig.register("custom_transformer", DecoderConfig)
        AutoModel.register(DecoderConfig, TransformerModelHF)
        AutoModelForCausalLM.register(DecoderConfig, TransformerModelHF)
        config = DecoderConfig(
            decoder_embed_dim=embed_dim,
            decoder_value_embed_dim=value_embed_dim,
            decoder_attention_heads=heads,
            decoder_ffn_embed_dim=ffn_dim,
            decoder_layers=layers,
            dropout=dropout,
            activation_dropout=activation_dropout,
            vocab_size=vocab_size,
            fsdp=fsdp,
            max_seq_len=seq_len)
        model = TransformerModelHF(config)

    # Print all arguments for recordkeeping
    print("Arguments:")
    arg_table = []
    row = []
    for i, arg in enumerate(arg_dict.keys()):
        row.append(f"{arg}: {arg_dict[arg]}")
        if (i + 1) % 4 == 0:
            arg_table.append(row)
            row = []
    if row:
        arg_table.append(row)
    print(tabulate(arg_table, tablefmt="grid"))

    # Print model info
    print("\nModel Summary:")
    total_params = model_summary(
        model,
        input_data=torch.ones(1, seq_len).long()).total_params

    # Create unique label for model (timestamp, model type, parameter count)
    model_label = f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}_" + \
        f"{model_type}_{total_params}"

    # Make sure dataset is pre-downloaded
    dataset_dir = Path(datasets_dir) / dataset_name
    assert dataset_dir.exists(), \
        f"The directory with data, {dataset_dir}, doesn't exist!"
    print(f"\nUsing dataset directory {dataset_dir}")

    # Initialize model directory for config files, weights, etc.
    model_dir = Path(data_dir) / "models" / model_label
    model_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving model files in {model_dir}")

    # Initialize weights directory
    weights_dir = model_dir / "weights"
    weights_dir.mkdir(parents=False, exist_ok=False)
    print(f"Saving weight files in {weights_dir}")

    # Initialize tokenizers directory
    tokenizers_dir = Path(data_dir) / "tokenizers"
    tokenizers_dir.mkdir(parents=False, exist_ok=True)
    print(f"Saving tokenizer files in {tokenizers_dir}")

    # Create SummaryWriter to record logs for TensorBoard
    if tboard_dir is None:
        tboard_log_dir = Path(data_dir) / "logs" / model_label
    else:
        tboard_log_dir = f"{tboard_dir}/logs/{model_label}"
    writer = SummaryWriter(log_dir=tboard_log_dir)
    print(f"Saving TensorBoard logs in {tboard_log_dir}")

    # Save all the variables in args as JSON inside folder
    json_string = json.dump(
        obj=arg_dict,
        fp=open(model_dir / "model_args.json", "w"),
        indent=4)

    # Print estimated loss if it hasn't learned anything
    print("\nEstimated Loss if guessing:")
    print(f"-log(1 / {vocab_size}) = " + \
        f"{-torch.log(torch.tensor(1 / vocab_size))}")
    
    # Get Tokenizer from local directory
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_folder)

    # Loads Tokenized data
    tokenized_train = load_ds(
        "parquet",
        data_files=str(Path(data_dir) / "tokenized_datasets" / dataset_name / "train.parquet"),
        split="all")
    tokenized_val = load_ds(
        "parquet",
        data_files=str(Path(data_dir) / "tokenized_datasets" / dataset_name / "validation.parquet"),
        split="all")
    tokenized_test = load_ds(
        "parquet",
        data_files=str(Path(data_dir) / "tokenized_datasets" / dataset_name / "test.parquet"),
        split="all")

    train_loader = DataLoader(
        tokenized_train.with_format("torch")["input_ids"],
        batch_size=batch_size,
        shuffle=True)
    valid_loader = DataLoader(
        tokenized_val.with_format("torch")["input_ids"],
        batch_size=batch_size)
    test_loader = DataLoader(
        tokenized_test.with_format("torch")["input_ids"],
        batch_size=batch_size)


    # Define loss function
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define the device to use
    device = torch.device(device)

    # Compile model and put on device
    model = torch.compile(model).to(device)

    # Train the model
    num_val_runs = 0
    for num_epoch in range(epochs):
        print(f"\nEpoch #{num_epoch}")

        model.train()
        train_total_loss = 0
        train_total_samples = 0
        for batch_idx, train_batch_seqs in enumerate(tqdm(
                train_loader,
                desc="Train")):
            # Put inputs and targets on device
            inputs = train_batch_seqs[:, :-1].to(device, non_blocking=True)
            targets = train_batch_seqs[:, 1:].to(device, non_blocking=True)

            # Zero out gradients
            optimizer.zero_grad()

            # Get model predictions
            predictions = model(inputs)

            # Reshape the model predictions for Cross Entropy
            predictions = predictions.transpose(-2, -1)

            # Calculate loss
            loss = loss_fn(predictions, targets)
            train_total_loss += loss * len(inputs)
            train_total_samples += len(inputs)

            # Backpropagate loss
            loss.backward()

            # Update parameters
            optimizer.step()

            # Run validation val_freq times per epoch. To do this, we split up
            # the epoch into val_freq chunks and run validation after each chunk
            # is finished.
            avg_val_loss = 0
            avg_train_loss = 0
            if val_freq > 0 \
                    and (num_val_runs + 1) / val_freq \
                        <= (batch_idx + 1) / len(train_loader):
                # Print average train loss
                avg_train_loss = train_total_loss / train_total_samples
                print("Average Train Loss Since Last Validation Run: " + \
                    f"{avg_train_loss}")
                train_total_loss = 0
                train_total_samples = 0

                print("Taking a break to run validation....")
                model.eval()
                val_total_loss = 0
                val_total_samples = 0
                with torch.inference_mode():
                    for val_batch_seqs in tqdm(valid_loader, desc="Validate"):
                        # Put validation inputs and targets on device
                        val_inputs = val_batch_seqs[:, :-1].to(
                            device,
                            non_blocking=True)
                        val_targets = val_batch_seqs[:, 1:].to(
                            device,
                            non_blocking=True)

                        # Get validation predictions
                        val_predictions = model(val_inputs)

                        # Reshape the model predictions for Cross Entropy
                        val_predictions = val_predictions.transpose(-2, -1)

                        # Calculate validation loss
                        val_loss = loss_fn(val_predictions, val_targets)
                        val_total_loss += val_loss.item() * len(val_inputs)
                        val_total_samples += len(val_inputs)

                    # Print average validation loss
                    avg_val_loss = val_total_loss / val_total_samples
                    print(f"\nAverage Validation Loss: {avg_val_loss}")

                # Log training and validation average loss
                writer.add_scalar(
                    tag="Loss/train",
                    scalar_value=avg_train_loss,
                    global_step=num_val_runs)
                writer.add_scalar(
                    tag="Loss/validation",
                    scalar_value=avg_val_loss,
                    global_step=num_val_runs)

                # If checkpoints are to be saved
                if checkpoints:
                    # Save current weights of the model
                    weight_filename = f"epoch_{num_epoch}_validation_" + \
                        f"{num_val_runs}.pt"
                    torch.save(
                        model.state_dict(),
                        weights_dir / weight_filename)
                    print(f"Saved weights as {weight_filename}")

                # Update how many validation runs there have been
                num_val_runs += 1

    # Test the model
    print("\nDone training! Now testing model...")
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.inference_mode():
        for test_batch_seqs in tqdm(test_loader, desc="Test"):
            # Put inputs and targets on device
            inputs = test_batch_seqs[:, :-1].to(device, non_blocking=True)
            targets = test_batch_seqs[:, 1:].to(device, non_blocking=True)

            # Get model predictions
            predictions = model(inputs)

            # Reshape the model predictions for Cross Entropy
            predictions = predictions.transpose(-2, -1)

            # Calculate loss
            loss = loss_fn(predictions, targets)
            total_loss += loss.item() * len(inputs)
            total_samples += len(inputs)

    # Print average testing loss
    avg_loss = total_loss / total_samples
    print(f"Average Test Loss: {avg_loss}")

    # Save hyperparameters and metrics in logs
    writer.add_hparams(
        hparam_dict=model.get_params(),
        metric_dict={
            "Loss/train": avg_train_loss,
            "Loss/validation": avg_val_loss,
            "Loss/test": avg_loss})

    # Close SummaryWriter
    writer.close()

    # Save completed model
    model.save_pretrained(model_dir)
    print(f"Saved completed model in {model_dir}")
    weight_filename = "training_completed.pt"
    torch.save(model.state_dict(), weights_dir / weight_filename)
    print(f"Saved final weights as {weight_filename}")

    return model, avg_loss


if __name__ == "__main__":
    # Initialize, setup, and parse the argument parser
    parser = ArgumentParser(
            prog="Model Trainer",
            description="Used to train comparable RetNet, Transformer models.")

    parser.add_argument("-a", "--activation-dropout", type=float, default=0.0,
        help="Probability of element to be zeroed in dropout layer after " + \
            "activation between FFN layers.")
    parser.add_argument("-b", "--batch-size", type=int, default=32,
        help="Batch size.")
    parser.add_argument("-c", "--checkpoints", action="store_true",
        default=False, help="Save model checkpoints while training.")
    parser.add_argument("--data-dir", type=str, required=True,
        help="Path to directory where all data except datasets are saved.")
    parser.add_argument("--dataset-name", type=str, default="wikitext",
        help="Hugging Face dataset name.")
    parser.add_argument("--datasets-dir", type=str, required=True,
        help="Path to the datasets directory.")
    parser.add_argument("--device", type=str, default="cuda",
        help="Device to use (ex: 'cpu', 'cuda', or 'cuda:0').")
    parser.add_argument("-d", "--dropout", type=float, default=0.1,
        help="Probability of element to be zeroed in dropout layer.")
    parser.add_argument("-e", "--embed-dim", type=int, default=768,
        help="Embedding dimension size of each token.")
    parser.add_argument("--epochs", type=int, default=10,
        help="Number of epochs to train for.")
    parser.add_argument("-f", "--ffn-dim", type=int, default=1280,
        help="FFN hidden layer size.")
    parser.add_argument("--fsdp", action="store_true", default=False,
        help="Module parameters sharded across data parallel workers.")
    parser.add_argument("-l", "--layers", type=int, default=12,
        help="Number of stacked layers in model.")
    parser.add_argument("--lr", type=float, required=True,
        help="Learning rate of model to train.")
    parser.add_argument("-m", "--model", required=True, dest="model_type",
        choices=["retnet", "transformer"],
        help="Name of model architecture to train.")
    parser.add_argument("-n", "--heads", type=int, default=3,
        help="Number of heads. Head architecture changes based on model.")
    parser.add_argument("-r", "--rand-seed", type=int, default=None,
        help="Random seed to use, allowing more reproducible results.")
    parser.add_argument("-s", "--seq-len", type=int, default=512,
        help="Sequence length (context window size).")
    parser.add_argument("--tboard-dir", type=str, default=None,
        help="Path to directory to save TensorBoard logs in.")
    parser.add_argument("--tokenizer-folder", type= str, required=True,
        help="Path to the file where the tokenizer will be saved")
    parser.add_argument("--val-freq", type=int, default=3,
        help="Number of times to run validation per epoch during training.")
    parser.add_argument("--value-embed-dim", type=int, default=1280,
        help="Value embed dimension size.")
    parser.add_argument("--vocab-size", type=int, required=True,
        help="Maximum number of unique tokens in vocabulary.")
    
    args = parser.parse_args()
    train_model(**vars(args))
