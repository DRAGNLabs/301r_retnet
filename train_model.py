import torch

from argparse import ArgumentParser
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder

if __name__ == "__main__":
    # Initialize, setup, and parse the argument parser
    parser = ArgumentParser(
            prog="Model Trainer",
            description="Used to train comparable RetNet, Transformer models.")

    parser.add_argument("-m", "--model", required=True,
            choices=["retnet", "transformer"],
            help="Name of model architecture to train.")
    parser.add_argument("--lr", type=float, required=True,
            help="Learning rate of model to train.")

    args = parser.parse_args()
    
