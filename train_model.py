import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import os.path
import json

from argparse import ArgumentParser
from datasets import load_wikitext2
from pathlib import Path
from tabulate import tabulate
from torch import Tensor
from torchinfo import summary as model_summary
from torchscale.architecture.config import DecoderConfig, RetNetConfig
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.retnet import RetNetDecoder
from tqdm import tqdm
from utils import generate_text

REPO_ROOT_NAME = "301r_retnet"

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
            checkpoint_activations: bool,
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
            checkpoint_activations (bool): Whether to perform checkpointing or
                not (done with the FairScale library).
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
                "checkpoint_activations": checkpoint_activations,
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
                checkpoint_activations=checkpoint_activations,
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
            checkpoint_activations: bool,
            fsdp: bool,
            max_seq_len: int):
        """ Use parameters to create corresponding RetNet model
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
            checkpoint_activations (bool): Whether to perform checkpointing or
                not (done with the FairScale library).
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
                "checkpoint_activations": checkpoint_activations,
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
                checkpoint_activations=checkpoint_activations,
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


if __name__ == "__main__":
    # Initialize, setup, and parse the argument parser
    parser = ArgumentParser(
            prog="Model Trainer",
            description="Used to train comparable RetNet, Transformer models.")

    parser.add_argument("-a", "--activation-dropout", type=float, default=0.0,
            help="Probability of element to be zeroed in dropout layer " + \
                    "after activation between FFN layers.")
    parser.add_argument("-c", "--checkpoint-activations", action="store_true",
            default=False, help="Use checkpointing.")
    parser.add_argument("-d", "--dropout", type=float, default=0.1,
            help="Probability of element to be zeroed in dropout layer.")
    parser.add_argument("-e", "--embed-dim", type=int, default=768,
            help="Embedding dimension size of each token.")
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
    parser.add_argument("-s", "--seq-len", type=int, default=512,
            help="Sequence length (context window size).")
    parser.add_argument("--value-embed-dim", type=int, default=1280,
            help="Value embed dimension size.")
    parser.add_argument("--vocab-size", type=int, required=True,
            help="Maximum number of unique tokens in vocabulary.")
    parser.add_argument("--batch-size", type=int, default=32,
            help="Batch size.")
    parser.add_argument("--device", type=str, default="cuda",
            help="Device to use (ex: 'cpu', 'cuda', or 'cuda:0').")
    parser.add_argument("--epochs", type=int, default=10,
            help="Number of epochs to train for.")

    args = parser.parse_args()

    # Test that the head dimension will be an even, whole number
    assert args.embed_dim % (args.heads * 2) == 0, \
            "Head Dimension must be even to perform Rotary Position " + \
            f"Embedding ({args.embed_dim} / {args.heads} = " + \
            f"{args.embed_dim / args.heads} -- not an even, whole number)! " + \
            "Try changing the Embedding Dimension or number of heads."

    # Test that the value embedding dimension is divisible by number of heads
    assert args.value_embed_dim % args.heads == 0, \
            "Value Embed Dimension not divisible by number of heads " + \
            f"({args.value_embed_dim} % {args.heads} != 0)!"

    # Create requested model
    if args.model == "retnet":
        model = RetNetModel(
                embed_dim=args.embed_dim,
                value_embed_dim=args.value_embed_dim,
                retention_heads=args.heads,
                ffn_dim=args.ffn_dim,
                layers=args.layers,
                dropout=args.dropout,
                activation_dropout=args.activation_dropout,
                vocab_size=args.vocab_size,
                checkpoint_activations=args.checkpoint_activations,
                fsdp=args.fsdp,
                max_seq_len=args.seq_len)
    elif args.model == "transformer":
        model = TransformerModel(
                embed_dim=args.embed_dim,
                value_embed_dim=args.value_embed_dim,
                attention_heads=args.heads,
                ffn_dim=args.ffn_dim,
                layers=args.layers,
                dropout=args.dropout,
                activation_dropout=args.activation_dropout,
                vocab_size=args.vocab_size,
                checkpoint_activations=args.checkpoint_activations,
                fsdp=args.fsdp,
                max_seq_len=args.seq_len)

    # Print all arguments for recordkeeping
    print("Arguments:")
    arg_table = []
    row = []
    for i, arg in enumerate(vars(args)):
        row.append(f"{arg}: {getattr(args, arg)}")
        if (i + 1) % 4 == 0:
            arg_table.append(row)
            row = []
    if row:
        arg_table.append(row)

    print(tabulate(arg_table, tablefmt="grid"))

    # Print model info
    print("\nModel Summary:")
    total_params = model_summary(model, input_data=torch.ones(1, args.seq_len)\
            .long()).total_params

    # Get path of repository root folder
    repo_root_dir = Path(__file__)
    while REPO_ROOT_NAME not in repo_root_dir.name:
        repo_root_dir = repo_root_dir.parent

    # Initialize model weights folders
    current_time = datetime.datetime.now()
    save_folder_dir = f"{current_time.strftime('%Y-%m-%d-%H:%M:%S')}_" + \
                      f"{args.model}_{total_params}"
    save_folder = repo_root_dir / "weights" / save_folder_dir
    save_folder.mkdir(parents=True, exist_ok=True)

    #Save all the variables in args as JSON inside folder
    arg_dict = vars(args)
    json_string = json.dump(obj=arg_dict,
                            fp=open(save_folder / "model_args.json", "w"),
                            indent=4)

    # Print estimated loss if it hasn't learned anything
    print("\nEstimated Loss if guessing:")
    print(f"-log(1 / {args.vocab_size}) = {-torch.log(torch.tensor(1 / args.vocab_size))}")

    # Get DataLoaders
    train_loader, valid_loader, test_loader, tokenizer = load_wikitext2(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            vocab_size=args.vocab_size)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Define the device to use
    device = torch.device(args.device)

    # Compile model and put on device
    model = torch.compile(model).to(device)

    # Train the model
    for num_epoch in range(args.epochs):
        print(f"Epoch #{num_epoch + 1}")

        model.train()
        train_total_loss = 0
        train_total_samples = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader,
                                                           desc="Train")):
            # Put inputs and targets on device
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

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

            # Run validation 3 times per epoch around 33%, 66%, and 100%
            if (3*(batch_idx + 1)/len(train_loader) % 1) <= 2/len(train_loader):
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
                    for val_inputs, val_targets in tqdm(valid_loader,
                                                        desc="Validate"):
                        # Put validation inputs and targets on device
                        val_inputs = val_inputs.to(device, non_blocking=True)
                        val_targets = val_targets.to(device, non_blocking=True)

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
        epoch_name = "model_" + str(num_epoch + 1) + ".pt"
        weights_epoch_file = os.path.join(save_folder, epoch_name)
        torch.save(model.state_dict(), weights_epoch_file)
        print(f"Saved weights in {weights_epoch_file}")

    # Test the model
    print("\nDone training! Now testing model...")
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.inference_mode():
        for inputs, targets in tqdm(test_loader, desc="Test"):
            # Put inputs and targets on device
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

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
            seq_len=args.seq_len,
            generation_length=100)

    print("Generated strings:")
    for idx, string in enumerate(generated_strings):
        print(f"{idx+1}: {string}\n")
