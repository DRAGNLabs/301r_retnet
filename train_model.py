import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

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
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    set_seed)
from utils import Struct

# Allow torch to run float32 matrix multiplications in lower precision for
# better performance while training if hardware is capable
torch.backends.cuda.matmul.allow_tf32 = True

def train_model(config: Struct):
    """ Use parameters to run train_model().

        Returns:
            A tuple of the trained model instance and the test average loss.
    """
    # Store all the parameters, which are the only locals at this point, as dict
    arg_dict = locals()

    # Test that the head dimension will be an even, whole number
    assert config.embed_dim % (config.heads * 2) == 0, \
        "Head Dimension must be even to perform Rotary Position Embedding " + \
        f"({config.embed_dim} / {config.heads} = " + \
        f"{config.embed_dim / config.heads} -- not an even, whole number)! " + \
        "Try changing the Embedding Dimension or number of heads."

    # Test that the value embedding dimension is divisible by number of heads
    assert config.value_embed_dim % config.heads == 0, \
        "Value Embed Dimension not divisible by number of heads " + \
        f"({config.value_embed_dim} % {config.heads} != 0)!"

    # Set random seeds for torch, numpy, random, etc. with transformers library
    if config.rand_seed is not None:
        set_seed(config.rand_seed)

    # Create requested model
    if config.model_type == "retnet":
        AutoConfig.register("retnet", RetNetConfig)
        AutoModel.register(RetNetConfig, RetNetModelHF)
        AutoModelForCausalLM.register(RetNetConfig, RetNetModelHF)
        HF_config = RetNetConfig(
            decoder_embed_dim=config.embed_dim,
            decoder_value_embed_dim=config.value_embed_dim,
            decoder_retention_heads=config.heads,
            decoder_ffn_embed_dim=config.ffn_dim,
            decoder_layers=config.layers,
            dropout=config.dropout,
            activation_dropout=config.activation_dropout,
            vocab_size=config.vocab_size,
            fsdp=config.fsdp,
            max_seq_len=config.seq_len)
        model = RetNetModelHF(HF_config)

    elif config.model_type == "transformer":
        AutoConfig.register("custom_transformer", DecoderConfig)
        AutoModel.register(DecoderConfig, TransformerModelHF)
        AutoModelForCausalLM.register(DecoderConfig, TransformerModelHF)
        HF_config = DecoderConfig(
            decoder_embed_dim=config.embed_dim,
            decoder_value_embed_dim=config.value_embed_dim,
            decoder_attention_heads=config.heads,
            decoder_ffn_embed_dim=config.ffn_dim,
            decoder_layers=config.layers,
            dropout=config.dropout,
            activation_dropout=config.activation_dropout,
            vocab_size=config.vocab_size,
            fsdp=config.fsdp,
            max_seq_len=config.seq_len)
        model = TransformerModelHF(HF_config)

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
        input_data=torch.ones(1, config.seq_len).long()).total_params

    # Create unique label for model (timestamp, model type, parameter count)
    model_label = f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}_" + \
        f"{config.model_type}_{total_params}"

    # Initialize model directory for config files, weights, etc.
    model_dir = Path(config.models_path) / model_label
    model_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving model files in {model_dir}")

    # Initialize weights directory
    weights_dir = model_dir / "weights"
    weights_dir.mkdir(parents=False, exist_ok=False)
    print(f"Saving weight files in {weights_dir}")

    # Create SummaryWriter to record logs for TensorBoard
    if config.tboard_path is None:
        tboard_log_dir = Path(config.models_path) / "logs" / model_label
    else:
        tboard_log_dir = f"{config.tboard_path}/logs/{model_label}"
    writer = SummaryWriter(log_dir=tboard_log_dir)
    print(f"Saving TensorBoard logs in {tboard_log_dir}")

    # Save all the variables in args as JSON inside folder
    json_string = json.dump(
        obj=arg_table,
        fp=open(model_dir / "model_args.json", "w"),
        indent=4)

    # Print estimated loss if it hasn't learned anything
    print("\nEstimated Loss if guessing:")
    print(f"-log(1 / {config.vocab_size}) = " + \
        f"{-torch.log(torch.tensor(1 / config.vocab_size))}")

    # Get Tokenizer from local directory
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_path)

    # Loads Tokenized data
    tokenized_train = load_ds(
        "parquet",
        data_files=str(Path(config.tokenized_dataset_path) / "train.parquet"),
        split="all")
    tokenized_val = load_ds(
        "parquet",
        data_files=str(Path(
            config.tokenized_dataset_path) / "validation.parquet"),
        split="all")
    tokenized_test = load_ds(
        "parquet",
        data_files=str(Path(config.tokenized_dataset_path) / "test.parquet"),
        split="all")

    train_loader = DataLoader(
        tokenized_train.with_format("torch")["input_ids"],
        batch_size=config.batch_size,
        shuffle=True)
    valid_loader = DataLoader(
        tokenized_val.with_format("torch")["input_ids"],
        batch_size=config.batch_size)
    test_loader = DataLoader(
        tokenized_test.with_format("torch")["input_ids"],
        batch_size=config.batch_size)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Define the device to use
    device = torch.device(config.device)

    # Compile model and put on device
    model = torch.compile(model).to(device)

    # Train the model
    num_val_runs = 0
    for num_epoch in range(config.epochs):
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
            if config.validation_frequency > 0 \
                    and (num_val_runs + 1) / config.validation_frequency \
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
                if config.checkpoints:
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
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    train_model(config)
