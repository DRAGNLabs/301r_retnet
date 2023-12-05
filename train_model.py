import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset

from argparse import ArgumentParser
from torchscale.architecture.config import DecoderConfig, RetNetConfig
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.retnet import RetNetDecoder

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torchinfo import summary as model_summary

class RetNetModel(nn.Module):
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
        """ Use parameters to create corresponding RetNet model
        Args:
            embed_dim (int): Dimension size of each embedded token.
            value_embed_dim (int): Value embed dimension size.
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
            max_seq_len (int): Size of context window.

        Returns:
            Created RetNetModel with given configuration.
        """
        super().__init__()

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

        # Save max_seq_len for padding later
        self.max_seq_len = max_seq_len

        # Save vocab_size for final dimensions later
        self.vocab_size = vocab_size

        # Create embeddings with index 0 representing padding
        self.text_embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=0)

        #TODO: Check that we are masking correctly
        self.decoder_stack = RetNetDecoder(config, embed_tokens=self.text_embeddings)

        # FFN after the final decoder
        #TODO: Double check we need this linear layer
        self.final_ffn = nn.Linear(
                in_features=max_seq_len * embed_dim,
                out_features=max_seq_len * vocab_size,
                bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add padding as needed to reach max_seq_len. Note that x comes with the
        # shape: (batch_size, seq_len)
        x = F.pad(
                input=x,
                pad=(0, self.max_seq_len - x.shape[-1]),
                mode="constant",
                value=0)

        # Last decoder results are in the shape:
        # (batch_size, max_seq_len, embed_dim)
        # TODO: Double check this is the output
        last_decoder_results = self.decoder_stack(x)[1]["inner_states"][-1]

        # Transform last decoder results to shape:
        # (batch_size, max_seq_len, vocab_size)
        # TODO: Double check if we need bias
        token_logits = self.final_ffn(
                torch.flatten(last_decoder_results, start_dim=1))\
                .reshape(-1, self.max_seq_len, self.vocab_size)

        # Return token predictions after Softmax activation
        # TODO: Check that they use softmax at the end for decoding
        return F.softmax(token_logits, dim=-1)


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
            dropout (float): Probability of an element to be zeroed during dropout.
            activation_dropout (float): Probability of an element to be zeroed
                during dropout after activation between FFN layers.
            vocab_size (int): Maximum vocabulary size (number of unique tokens in
                vocabulary.
            checkpoint_activations (bool): Whether to perform checkpointing or not
                (done with the FairScale library).
            fsdp (bool): Whether to shard Module parameters across data parallel
                workers or not (with the FairScale library).
            max_seq_len (int): Size of context window.

        Returns:
            Created TransformerModel with given configuration.
        """
        super().__init__()

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

        # Save max_seq_len for padding later
        self.max_seq_len = max_seq_len

        # Save vocab_size for final dimensions later
        self.vocab_size = vocab_size

        # Create embeddings with index 0 representing padding
        self.text_embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=0)

        self.decoder_stack = Decoder(config, embed_tokens=self.text_embeddings)

        # FFN after the final decoder
        self.final_ffn = nn.Linear(
                in_features=max_seq_len * embed_dim,
                out_features=max_seq_len * vocab_size,
                bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add padding as needed to reach max_seq_len. Note that x comes with the
        # shape: (batch_size, seq_len)
        x = F.pad(
                input=x,
                pad=(0, self.max_seq_len - x.shape[-1]),
                mode="constant",
                value=0)

        # Last decoder results are in the shape:
        # (batch_size, max_seq_len, embed_dim)
        last_decoder_results = self.decoder_stack(x)[1]["inner_states"][-1]

        # Transform last decoder results to shape:
        # (batch_size, max_seq_len, vocab_size)
        token_logits = self.final_ffn(
                torch.flatten(last_decoder_results, start_dim=1))\
                .reshape(-1, self.max_seq_len, self.vocab_size)

        # Return token predictions after Softmax activation
        return F.softmax(token_logits, dim=-1)

def load_dataset():
    # This webpage shows a way to read torchtext into a Dataloader
    # use their tokenizer for now to get training going
    # : https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    # TODO: Use sentencepice tokenizer in the future
    

    return None, None, None, None, None


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
    parser.add_argument("-s", "--seq-len", type=int, default=512,
            help="Sequence length (context window size).")
    parser.add_argument("--value-embed-dim", type=int, default=1280,
            help="Value embed dimension size.")
    parser.add_argument("--vocab-size", type=int, required=True,
            help="Maximum number of unique tokens in vocabulary.")

    args = parser.parse_args()
    
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
    else:
        raise ValueError("Model name not recognized.")
    
    # Print model info
    model_summary(model, input_data=torch.ones(1, args.seq_len).long())

    # Load the dataset
    batch_size, eval_batch_size, train_data, val_data, test_data  = load_dataset()

    # Train the model
    print('test')
