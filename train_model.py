import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from datetime import datetime
from load_data import get_loaders_tokenizer
from math import isclose
from pathlib import Path
from tabulate import tabulate
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary as model_summary
from torchscale.architecture.config import DecoderConfig, RetNetConfig
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.retnet import RetNetDecoder
from tqdm import tqdm
from transformers import set_seed
from utils import generate_text

# Allow torch to run float32 matrix multiplications in lower precision for
# better performance while training if hardware is capable
torch.backends.cuda.matmul.allow_tf32 = True

class RetNetModel(nn.Module):
    """ Create model with RetNet architecture. """
    def __init__(
            self,
            embed_dim: int,
            value_embed_dim: int,
            retention_heads: int,
            ffn_dim: int,
            layers: int,
            dropout: float,
            activation_dropout: float,
            vocab_size: int,
            fsdp: bool,
            max_seq_len: int):
        """ Use parameters to create corresponding RetNet model.
        Args:
            embed_dim (int): Dimension size of each embedded token.
            value_embed_dim (int): Value embed dimension size.
            retention_heads (int): Number of retention heads in MSR module.
            ffn_dim (int): Hidden layer size of Feed Forward Network (FFN).
            layers (int): Number of retention network layers.
            dropout (float): Probability of an element to be zeroed during
                dropout.
            activation_dropout (float): Probability of an element to be zeroed
                during dropout after activation between FFN layers.
            vocab_size (int): Maximum vocabulary size (number of unique tokens
                in vocabulary.
            fsdp (bool): Whether to shard Module parameters across data parallel
                workers or not (with the FairScale library).
            max_seq_len (int): Size of context window.
        """
        super().__init__()

        # Store hyperparameters
        self.model_params = {
            "embed_dim": embed_dim,
            "value_embed_dim": value_embed_dim,
            "retention_heads": retention_heads,
            "ffn_dim": ffn_dim,
            "layers": layers,
            "dropout": dropout,
            "activation_dropout": activation_dropout,
            "vocab_size": vocab_size,
            "fsdp": fsdp,
            "max_seq_len": max_seq_len}

        # Create RetNet configuration
        config = RetNetConfig(
            decoder_embed_dim=embed_dim,
            decoder_value_embed_dim=value_embed_dim,
            decoder_retention_heads=retention_heads,
            decoder_ffn_embed_dim=ffn_dim,
            decoder_layers=layers,
            dropout=dropout,
            activation_dropout=activation_dropout,
            vocab_size=vocab_size,
            fsdp=fsdp)

        # Create embeddings with index 0 representing padding
        text_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0)

        self.decoder_stack = RetNetDecoder(config, embed_tokens=text_embeddings)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Long tensor of dimensions: (batch size, sequence
                length).

        Returns:
            A tensor of dimensions: (batch size, sequence length, vocabulary
                size).
        """
        preds, _ = self.decoder_stack(x)
        return preds

    def get_params(self) -> dict:
        """ Get model parameters dictionary. """
        return self.model_params


class TransformerModel(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            value_embed_dim: int,
            attention_heads: int,
            ffn_dim: int,
            layers: int,
            dropout: float,
            activation_dropout: float,
            vocab_size: int,
            fsdp: bool,
            max_seq_len: int):
        """ Use parameters to create corresponding Transformer model.
        Args:
            embed_dim (int): Dimension size of each embedded token.
            value_embed_dim (int): Value embed dimension size.
            attention_heads (int): Number of attention heads in MHA module.
            ffn_dim (int): Hidden layer size of Feed Forward Network (FFN).
            layers (int): Number of retention network layers.
            dropout (float): Probability of an element to be zeroed during
                dropout.
            activation_dropout (float): Probability of an element to be zeroed
                during dropout after activation between FFN layers.
            vocab_size (int): Maximum vocabulary size (number of unique tokens
                in vocabulary.
            fsdp (bool): Whether to shard Module parameters across data parallel
                workers or not (with the FairScale library).
            max_seq_len (int): Size of context window.
        """
        super().__init__()

        # Store hyperparameters
        self.model_params = {
            "embed_dim": embed_dim,
            "value_embed_dim": value_embed_dim,
            "attention_heads": attention_heads,
            "ffn_dim": ffn_dim,
            "layers": layers,
            "dropout": dropout,
            "activation_dropout": activation_dropout,
            "vocab_size": vocab_size,
            "fsdp": fsdp,
            "max_seq_len": max_seq_len}

        # Create Transformer Decoder configuration
        config = DecoderConfig(
            decoder_embed_dim=embed_dim,
            decoder_value_embed_dim=value_embed_dim,
            decoder_attention_heads=attention_heads,
            decoder_ffn_embed_dim=ffn_dim,
            decoder_layers=layers,
            dropout=dropout,
            activation_dropout=activation_dropout,
            vocab_size=vocab_size,
            fsdp=fsdp)

        # Create embeddings with index 0 representing padding
        text_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0)

        self.decoder_stack = Decoder(config, embed_tokens=text_embeddings)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Long tensor of dimensions: (batch size, sequence
                length).

        Returns:
            A tensor of dimensions: (batch size, sequence length, vocabulary
                size).
        """
        preds, _ = self.decoder_stack(x)
        return preds

    def get_params(self) -> dict:
        """ Get model parameters dictionary. """
        return self.model_params


def train_model(activation_dropout=0.0, batch_size=8, checkpoints=False, 
                data_dir=None, dataset_feature=None, dataset_name="wikitext", dataset_subset="wikitext-2-v1", device="cuda",
         dropout=0.1, embed_dim=76, epochs=1, ffn_dim=12, fsdp=False, heads=4, 
         layers=2, lr=0.001, model_type="retnet", rand_seed=None, repo_root_dir=None,
         seq_len=128, splits=[0.7, 0.2, 0.1], tboard_dir=None, val_freq=3, value_embed_dim=12, vocab_size=4000):
    arg_dict = locals()
    print(arg_dict)

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

    # Test the dataset splits add up to 1, using isclose for rounding errors
    assert isclose(sum(splits), 1), \
        "The dataset splits for the training, validation, and testing " + \
        f"datasets must sum up to 1 ({' + '.join(map(str, splits))} != 1)!"

    # Set random seeds for torch, numpy, random, etc. with transformers library
    if rand_seed is not None:
        set_seed(rand_seed)

    # Create requested model
    if model_type == "retnet":
        model = RetNetModel(
                embed_dim=embed_dim,
                value_embed_dim=value_embed_dim,
                retention_heads=heads,
                ffn_dim=ffn_dim,
                layers=layers,
                dropout=dropout,
                activation_dropout=activation_dropout,
                vocab_size=vocab_size,
                fsdp=fsdp,
                max_seq_len=seq_len)
    elif model_type == "transformer":
        model = TransformerModel(
                embed_dim=embed_dim,
                value_embed_dim=value_embed_dim,
                attention_heads=heads,
                ffn_dim=ffn_dim,
                layers=layers,
                dropout=dropout,
                activation_dropout=activation_dropout,
                vocab_size=vocab_size,
                fsdp=fsdp,
                max_seq_len=seq_len)
        
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
    dataset_root_dir = Path(dataset_dir)
    dataset_dir = dataset_root_dir / dataset_name
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
        tboard_log_folder = repo_root_dir / "logs" / model_label
    else:
        tboard_log_folder = f"{tboard_dir}/logs/{model_label}"
    writer = SummaryWriter(log_dir=tboard_log_folder)
    print(f"Saving TensorBoard logs in {tboard_log_folder}")

    # Save all the variables in args as JSON inside folder
    arg_dict = vars(args)
    json_string = json.dump(
        obj=arg_dict,
        fp=open(model_dir / "model_args.json", "w"),
        indent=4)

    # Print estimated loss if it hasn't learned anything
    print("\nEstimated Loss if guessing:")
    print(f"-log(1 / {vocab_size}) = {-torch.log(torch.tensor(1 / vocab_size))}")

    # Get DataLoaders and trained Tokenizer
    print(f"\nNow retrieving '{dataset_name}' and training tokenizer...")
    train_loader, valid_loader, test_loader, tokenizer = get_loaders_tokenizer(
        dataset_name=dataset_name,
        seq_len=seq_len,
        batch_size=batch_size,
        vocab_size=vocab_size,
        dataset_dir=dataset_dir,
        dataset_config=dataset_subset,
        text_feature=dataset_feature,
        max_token_len=20,
        splits=splits,
        rand_seed=rand_seed)

    # Save trained tokenizer
    tokenizer.save_pretrained(
        save_directory=tokenizers_dir,
        filename_prefix="BPE")
    print(f"Saved trained tokenizer in {tokenizers_dir}")

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

            # Run validation val_freq times per epoch. To do this, we split
            # up the epoch into val_freq chunks and run validation after
            # each chunk is finished.
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
    weight_filename = "training_completed.pt"
    torch.save(model.state_dict(), weights_dir / weight_filename)
    print(f"Saved final weights as {weight_filename}")

    # Generate text from the model
    print("\nGenerating text...")
    input_starting_strings = [
        "<pad>",
        "= valkyria",
        "= = reception ="]

    generated_strings = generate_text(
        model=model,
        tokenizer=tokenizer,
        start_string_list=input_starting_strings,
        device=device,
        seq_len=seq_len,
        generation_length=100)

    print("Generated strings:")
    for idx, string in enumerate(generated_strings):
        print(f"{idx+1}: {string}\n")
        
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
    parser.add_argument("--dataset-dir", type=str, required=True,
        help="Path to directory in which Hugging Face datasets are downloaded.")
    parser.add_argument("--dataset-feature", type=str, default="text",
        help="Hugging Face dataset feature/column to use.")
    parser.add_argument("--dataset-name", type=str, default="wikitext",
        help="Hugging Face dataset name. Should also set --dataset-subset.")
    parser.add_argument("--dataset-subset", type=str, default="wikitext-2-v1",
        help="Subset/config to use for Hugging Face dataset.")
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
    parser.add_argument("-m", "--model", required=True,
        choices=["retnet", "transformer"],
        help="Name of model architecture to train.")
    parser.add_argument("-n", "--heads", type=int, default=3,
        help="Number of heads. Head architecture changes based on model.")
    parser.add_argument("-r", "--rand-seed", type=int, default=None,
        help="Random seed to use, allowing more reproducible results.")
    parser.add_argument("-s", "--seq-len", type=int, default=512,
        help="Sequence length (context window size).")
    parser.add_argument("--splits", type=float, nargs=3,
        default=[0.7, 0.2, 0.1],
        help="Space-separated decimal splits of train, validation, and " + \
            "test datasets. (Ex: '0.7 0.2 0.1')")
    parser.add_argument("--tboard-dir", type=str, default=None,
        help="Path to directory to save TensorBoard logs in.")
    parser.add_argument("--val-freq", type=int, default=3,
        help="Number of times to run validation per epoch during training.")
    parser.add_argument("--value-embed-dim", type=int, default=1280,
        help="Value embed dimension size.")
    parser.add_argument("--vocab-size", type=int, required=True,
        help="Maximum number of unique tokens in vocabulary.")

    args = parser.parse_args()

    train_model(
        activation_dropout=args.activation_dropout, 
        batch_size=args.batch_size, 
        checkpoints=args.checkpoints, 
        data_dir=args.data_dir,
        dataset_feature=args.dataset_feature,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        device=args.device, 
        dropout=args.dropout, 
        embed_dim=args.embed_dim, 
        epochs=args.epochs, 
        ffn_dim=args.ffn_dim, 
        fsdp=args.fsdp, 
        heads=args.heads, 
        layers=args.layers, 
        lr=args.lr, 
        model_type=args.model, 
        rand_seed=args.rand_seed, 
        seq_len=args.seq_len, 
        splits=args.splits,
        tboard_dir=args.tboard_dir,
        val_freq=args.val_freq, 
        value_embed_dim=args.value_embed_dim, 
        vocab_size=args.vocab_size
    )
