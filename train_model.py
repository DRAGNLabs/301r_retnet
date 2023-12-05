import torch

from argparse import ArgumentParser
from torchscale.architecture.config import DecoderConfig, RetNetConfig
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.retnet import RetNetDecoder

def get_retnet_model(
        embed_dim: int,
        retention_heads: int,
        ffn_dim: int,
        layers: int,
        dropout: float,
        activation_dropout: float,
        vocab_size: int,
        checkpoint_activations: bool,
        fsdp: bool) -> RetNetDecoder:
    """ Use parameters to create corresponding RetNet model
    Args:
        embed_dim (int): Dimension size of each embedded token.
        retention_heads (int): Number of retention heads in MSR module.
        ffn_dim (int): Hidden layer size of Feed Forward Network (FFN).
        layers (int): Number of retention network layers.
        dropout (float): Probability of an element to be zeroed during dropout.
        activation_dropout (float): Probability of an element to be zeroed
            during dropout after activation between FFN layers.
        vocab_size (int): Maximum vocabulary size (number of unique tokens in
            vocabulary.
        checkpoint_activations (bool): Whether to perform checkpointing or not
            (done with the FairScale library).
        fsdp (bool): Whether to shard Module parameters across data parallel
            workers or not (with the FairScale library).

    Returns:
        Created RetNetDecoder with given configuration.
    """
    config = RetNetConfig(
            decoder_embed_dim=embed_dim,
            decoder_retention_heads=retention_heads,
            decoder_ffn_embed_dim=ffn_dim,
            decoder_layers=layers,
            dropout=dropout,
            activation_dropout=activation_dropout,
            vocab_size=vocab_size,
            checkpoint_activations=checkpoint_activations,
            fsdp=fsdp)

    return RetNetDecoder(config)


def get_transformer_model(
        embed_dim: int,
        attention_heads: int,
        ffn_dim: int,
        layers: int,
        dropout: float,
        activation_dropout: float,
        vocab_size: int,
        checkpoint_activations: bool,
        fsdp: bool) -> Decoder:
    """ Use parameters to create corresponding Transformer model
    Args:
        embed_dim (int): Dimension size of each embedded token.
        attention_heads (int): Number of attention heads in MHA module.
        ffn_dim (int): Hidden layer size of Feed Forward Network (FFN).
        layers (int): Number of retention network layers.
        dropout (float): Probability of an element to be zeroed during dropout.
        activation_dropout (float): Probability of an element to be zeroed
            during dropout after activation between FFN layers.
        vocab_size (int): Maximum vocabulary size (number of unique tokens in
            vocabulary.
        checkpoint_activations (bool): Whether to perform checkpointing or not
            (done with the FairScale library).
        fsdp (bool): Whether to shard Module parameters across data parallel
            workers or not (with the FairScale library).

    Returns:
        Created Decoder with given configuration.
    """
    config = DecoderConfig(
            decoder_embed_dim=embed_dim,
            decoder_attention_heads=attention_heads,
            decoder_ffn_embed_dim=ffn_dim,
            decoder_layers=layers,
            dropout=dropout,
            activation_dropout=activation_dropout,
            vocab_size=vocab_size,
            checkpoint_activations=checkpoint_activations,
            fsdp=fsdp)

    return Decoder(config)


if __name__ == "__main__":
    # Initialize, setup, and parse the argument parser
    parser = ArgumentParser(
            prog="Model Trainer",
            description="Used to train comparable RetNet, Transformer models.")

    parser.add_argument("-a", "--activation-dropout", type=float, default=0.0,
            help="Probability of element to be zeroed in dropout layer " + \
                    "after activation between FFN layers.")
    parser.add_argument("-c", "--checkpoint-activations", type=bool,
            default=False, help="Use checkpointing.")
    parser.add_argument("-d", "--dropout", type=float, default=0.0,
            help="Probability of element to be zeroed in dropout layer.")
    parser.add_argument("-e", "--embed-dim", type=int, default=768,
            help="Embedding dimension size of each token.")
    parser.add_argument("-f", "--ffn-dim", type=int, default=1280,
            help="FFN hidden layer size.")
    parser.add_argument("--fsdp", type=bool, default=False,
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
    parser.add_argument("--vocab-size", type=int, required=True,
            help="Maximum number of unique tokens in vocabulary.")

    args = parser.parse_args()
    
    # Create requested model
    if args.model == "retnet":
        model = get_retnet_model(
                embed_dim=args.embed_dim,
                retention_heads=args.heads,
                ffn_dim=args.ffn_dim,
                layers=args.layers,
                dropout=args.dropout,
                activation_dropout=args.activation_dropout,
                vocab_size=args.vocab_size,
                checkpoint_activations=args.checkpoint_activations,
                fsdp=args.fsdp)
    elif args.model == "transformer":
        model = get_transformer_model(
                embed_dim=args.embed_dim,
                attention_heads=args.heads,
                ffn_dim=args.ffn_dim,
                layers=args.layers,
                dropout=args.dropout,
                activation_dropout=args.activation_dropout,
                vocab_size=args.vocab_size,
                checkpoint_activations=args.checkpoint_activations,
                fsdp=args.fsdp)
