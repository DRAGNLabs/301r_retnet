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
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary as model_summary
from tqdm import tqdm
from transformers import set_seed
from utils import generate_text
from hugging_face_model import RetNetModel, TransformerModel

REPO_ROOT_NAME = "301r_retnet"

# Allow torch to run float32 matrix multiplications in lower precision for
# better performance while training if hardware is capable
torch.backends.cuda.matmul.allow_tf32 = True

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
    parser.add_argument("--val-freq", type=int, default=3,
        help="Number of times to run validation per epoch during training.")
    parser.add_argument("--value-embed-dim", type=int, default=1280,
        help="Value embed dimension size.")
    parser.add_argument("--vocab-size", type=int, required=True,
        help="Maximum number of unique tokens in vocabulary.")

    args = parser.parse_args()

    # Test that the head dimension will be an even, whole number
    assert args.embed_dim % (args.heads * 2) == 0, \
        "Head Dimension must be even to perform Rotary Position Embedding " + \
        f"({args.embed_dim} / {args.heads} = {args.embed_dim / args.heads} " + \
        "-- not an even, whole number)! Try changing the Embedding " + \
        "Dimension or number of heads."

    # Test that the value embedding dimension is divisible by number of heads
    assert args.value_embed_dim % args.heads == 0, \
        "Value Embed Dimension not divisible by number of heads " + \
        f"({args.value_embed_dim} % {args.heads} != 0)!"

    # Test the dataset splits add up to 1, using isclose for rounding errors
    assert isclose(sum(args.splits), 1), \
        "The dataset splits for the training, validation, and testing " + \
        f"datasets must sum up to 1 ({' + '.join(map(str, args.splits))} != 1)!"

    # Set random seeds for torch, numpy, random, etc. with transformers library
    if args.rand_seed is not None:
        set_seed(args.rand_seed)

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
            fsdp=args.fsdp,
            max_seq_len=args.seq_len)
        model.config.save_pretrained('retnet_config')
        torch.save(model.state_dict(), "./retnet_config/retnet301.pt")
        
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
            fsdp=args.fsdp,
            max_seq_len=args.seq_len)
        model.config.save_pretrained('transformer_config')
        torch.save(model.state_dict(), "transformer_config/transformer.pt")
        

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
    total_params = model_summary(
        model,
        input_data=torch.ones(1, args.seq_len).long()).total_params

    # Get path of repository root folder
    repo_root_dir = Path(__file__)
    while REPO_ROOT_NAME not in repo_root_dir.name:
        repo_root_dir = repo_root_dir.parent

    # Create unique label for model (timestamp, model type, parameter count)
    model_label = f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}_" + \
        f"{args.model}_{total_params}"

    # Initialize model weights folders
    save_folder = repo_root_dir / "weights" / model_label
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving weights in {save_folder}")

    # Save all the variables in args as JSON inside folder
    arg_dict = vars(args)
    json_string = json.dump(
        obj=arg_dict,
        fp=open(save_folder / "model_args.json", "w"),
        indent=4)

    # Create SummaryWriter to record logs for TensorBoard
    writer = SummaryWriter(log_dir=repo_root_dir / "logs" / model_label)

    # Print estimated loss if it hasn't learned anything
    print("\nEstimated Loss if guessing:")
    print(f"-log(1 / {args.vocab_size}) = " + \
        f"{-torch.log(torch.tensor(1 / args.vocab_size))}")

    # Get DataLoaders and trained Tokenizer
    print(f"\nNow retrieving '{args.dataset_name}' and training tokenizer...")
    train_loader, valid_loader, test_loader, tokenizer = get_loaders_tokenizer(
        dataset_name=args.dataset_name,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        data_dir=repo_root_dir / "data",
        dataset_config=args.dataset_subset,
        text_feature=args.dataset_feature,
        max_token_len=20,
        splits=args.splits,
        rand_seed=args.rand_seed)

    # Save trained tokenizer
    tokenizer.save_pretrained(save_directory=save_folder, filename_prefix="BPE")
    print(f"Saved trained tokenizer")

    # Define loss function
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Define the device to use
    device = torch.device(args.device)

    # Compile model and put on device
    model = torch.compile(model).to(device)

    # Train the model
    num_val_runs = 0
    for num_epoch in range(args.epochs):
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

            # Run validation args.validation_freq times per epoch. To do this,
            # we split up the epoch into arg.validation_freq chunks and run
            # validation after each chunk is finished.
            progress_through_chunk = args.val_freq * (batch_idx + 1) \
                                     / len(train_loader) % 1
            if progress_through_chunk <= (args.val_freq-1) / len(train_loader):
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
                if args.checkpoints:
                    # Save current weights of the model
                    weight_filename = f"epoch_{num_epoch}_validation_" + \
                        f"{num_val_runs}.pt"
                    torch.save(
                        model.state_dict(),
                        save_folder / weight_filename)
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
    torch.save(model.state_dict(), save_folder / weight_filename)
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
        seq_len=args.seq_len,
        generation_length=100)

    print("Generated strings:")
    for idx, string in enumerate(generated_strings):
        print(f"{idx+1}: {string}\n")
