import math

import numpy as np
import torch
import torch.nn as nn
from fairscale.nn import checkpoint_wrapper, wrap

from torchscale.component.performer.attention import SelfAttention, CrossAttention
from torchscale.component.performer.attention import FixedPositionalEmbedding, AbsolutePositionalEmbedding
from torchscale.component.performer.axial_positional_embedding import AxialPositionalEmbedding

from torchscale.architecture.utils import init_bert_params
from torchscale.component.droppath import DropPath
from torchscale.component.feedforward_network import FeedForwardNetwork, make_experts
from torchscale.component.relative_position_bias import RelativePositionBias
from torchscale.component.xmoe.moe_layer import MOELayer
from torchscale.component.xmoe.routing import Top1Gate, Top2Gate
try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm


# The DecoderLayer class represents a single layer of the Performer Decoder.
class DecoderLayer(nn.Module):
    def __init__(
        self,
        args,
        depth,
        is_moe_layer=False,
        is_encoder_decoder=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = torch.nn.Dropout(args.dropout)

        if args.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, args.drop_path_rate, args.decoder_layers)[
                depth
            ]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        # Performer self-attention mechanism which calls fast attention
        self.self_attn = self.build_self_attention(self.embed_dim, args)

        self.normalize_before = args.decoder_normalize_before

        # Layer normalization applied before the self-attention and feedforward network (Pre-LN transformer).
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, eps=args.layernorm_eps)

        # If this is part of an encoder-decoder architecture, set up the attention mechanism for encoder outputs.
        if not is_encoder_decoder:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, eps=args.layernorm_eps)

        self.is_moe_layer = is_moe_layer
        self.ffn_dim = args.decoder_ffn_embed_dim

        # Feedforward network as defined in the transformer model
        if not self.is_moe_layer:
            self.ffn = self.build_ffn(
                self.embed_dim,
                self.args,
            )
        else:
            if args.moe_top1_expert:
                gate = Top1Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    use_fp32=args.moe_gating_use_fp32,
                    moe_eval_capacity_token_fraction=args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    args.moe_gating_use_fp32,
                    args.moe_second_expert_policy,
                    args.moe_normalize_gate_prob_before_dropping,
                    args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            experts = make_experts(args, self.embed_dim, self.ffn_dim)
            self.moe_layer = MOELayer(gate, experts, args)

         # The final layer normalization applied after the self-attention and feedforward network.
        self.final_layer_norm = LayerNorm(self.embed_dim, eps=args.layernorm_eps)

         # DeepNorm initialization strategy for stable training of deep transformer networks.
        if args.deepnorm:
            if is_encoder_decoder:
                self.alpha = math.pow(3.0 * args.decoder_layers, 0.25)
            else:
                self.alpha = math.pow(2.0 * args.decoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def build_ffn(self, embed_dim, args):
        return FeedForwardNetwork(
            embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
            args.layernorm_eps,
            args.subln,
        )

    def build_self_attention(self, embed_dim, args):
        # Replace with FastAttention initialization
        return SelfAttention(
            dim=args.embed_dim,
            heads=args.decoder_attention_heads,
            causal=True,  # Enabling causal attention
            dim_head=64,  # Example dimension, adjust as necessary
            local_heads = 0,
            local_window_size = 256,
            nb_features = None, # Defaults to int(dim_head * log(dim_head))
            feature_redraw_interval = 1000,
            generalized_attention = False,
            kernel_fn = nn.ReLU(),
            dropout=args.dropout,
            no_projection = False,
            qkv_bias = False,
            attn_out_bias = True
        )

    def build_encoder_attention(self, embed_dim, args):
        return CrossAttention(
            dim=args.embed_dim,
            heads=args.decoder_attention_heads,
            causal=False,
            dim_head=64,
            local_heads = 0,
            local_window_size = 256,
            nb_features = None,
            feature_redraw_interval = 1000,
            generalized_attention = False,
            kernel_fn = nn.ReLU(),
            dropout=args.dropout,
            no_projection = False,
            qkv_bias = False,
            attn_out_bias = True
        )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    # The forward method defines the operations to process the input through this layer.
    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        self_attn_rel_pos=None,
        cross_attn_rel_pos=None,
        is_first_step=False,
    ):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Apply performer self-attention. The attention mask ensures causality in the decoder.
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            attn_mask=self_attn_mask,
            rel_pos=self_attn_rel_pos,
            is_first_step=is_first_step,
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=None,
                rel_pos=cross_attn_rel_pos,
            )
            x = self.dropout_module(x)

            if self.drop_path is not None:
                x = self.drop_path(x)

            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer:
            x = self.ffn(x)
            l_aux = None
        else:
            x, l_aux = self.moe_layer(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn, None, l_aux


# The PerformerDecoder class represents the entire decoder made up of multiple layers of DecoderLayer.
class PerformerDecoder(nn.Module):
    # Initialization sets up the embedding layers and stacks of DecoderLayer modules.
    def __init__(
        self,
        args,
        embed_tokens=None,
        embed_positions=None,
        output_projection=None,
        is_encoder_decoder=False,
        max_seq_len,
        rotary_position_emb = True,
        axial_position_emb = False,
        axial_position_shape = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.args = args

        self.dropout_module = torch.nn.Dropout(args.dropout)

        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        # Token embeddings convert input tokens into vector representations.
        self.embed_tokens = embed_tokens
        # Positional embeddings provide the model with information about the token positions.
        self.embed_positions = embed_positions

        # Performer positional embedding initialization
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)

        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif axial_position_emb:
            axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / 64), 64))
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)

        if (
            output_projection is None
            and not args.no_output_layer
            and args.vocab_size > 0
        ):
            self.output_projection = self.build_output_projection(args)
        else:
            self.output_projection = output_projection

        if args.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, eps=args.layernorm_eps)
        else:
            self.layernorm_embedding = None

        # Initialize and stack the DecoderLayer instances.
        self.layers = nn.ModuleList([])

        moe_freq = args.moe_freq
        for i in range(args.decoder_layers):
            is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
            self.layers.append(
                self.build_decoder_layer(
                    args,
                    depth=i,
                    is_moe_layer=is_moe_layer,
                    is_encoder_decoder=is_encoder_decoder,
                )
            )

        self.num_layers = len(self.layers)

        if args.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim, eps=args.layernorm_eps)
        else:
            self.layer_norm = None

        self.self_attn_relative_position = None
        self.cross_attn_relative_position = None

        if args.rel_pos_buckets > 0 and args.max_rel_pos > 0:
            self.self_attn_relative_position = RelativePositionBias(
                num_buckets=args.rel_pos_buckets,
                max_distance=args.max_rel_pos,
                n_heads=args.decoder_attention_heads,
            )
            if is_encoder_decoder:
                self.cross_attn_relative_position = RelativePositionBias(
                    num_buckets=args.rel_pos_buckets,
                    max_distance=args.max_rel_pos,
                    n_heads=args.decoder_attention_heads,
                )

        if args.bert_init:
            self.apply(init_bert_params)

        if args.deepnorm:
            if is_encoder_decoder:
                init_scale = math.pow(12.0 * args.decoder_layers, 0.25)
            else:
                init_scale = math.pow(8.0 * args.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

        if args.subln:
            if is_encoder_decoder:
                init_scale = math.sqrt(math.log(args.decoder_layers * 3))
            else:
                init_scale = math.sqrt(math.log(args.decoder_layers * 2))
            for name, p in self.named_parameters():
                if "encoder_attn" in name:
                    continue
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.mul_(init_scale)

    def build_output_projection(
        self,
        args,
    ):
        if args.share_decoder_input_output_embed:
            output_projection = torch.nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = self.embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(
                args.decoder_embed_dim, args.vocab_size, bias=False
            )
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.decoder_embed_dim**-0.5
            )
        return output_projection

    def build_decoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = DecoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer

    # The forward_embedding method applies embeddings and position encodings to the input tokens.
    def forward_embedding(
        self,
        tokens,
        token_embedding=None,
        incremental_state=None,
    ):
        # positions = None
        # if self.embed_positions is not None:
        #     positions = self.embed_positions(
        #         tokens, incremental_state=incremental_state
        #     )

        # if incremental_state is not None and not self.is_first_step(incremental_state):
        #     tokens = tokens[:, -1:]
        #     if positions is not None:
        #         positions = positions[:, -1:]

        # if token_embedding is None:
        #     token_embedding = self.embed_tokens(tokens)

        # x = embed = self.embed_scale * token_embedding

        # if positions is not None:
        #     x += positions

        # if self.layernorm_embedding is not None:
        #     x = self.layernorm_embedding(x)

        # x = self.dropout_module(x)


        # token and positional embeddings
        x = self.token_emb(x)
        x += self.pos_emb(x)

        x = self.dropout_module(x)

        embed = x

        return x, embed

    def is_first_step(self, incremental_state):
        if incremental_state is None:
            return False
        return incremental_state.get("is_first_step", False)

    # The forward method processes the input tokens through the entire decoder.
    def forward(
        self,
        prev_output_tokens,
        self_attn_padding_mask=None,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        return_all_hiddens=False,
        token_embeddings=None,
        **kwargs
    ):
        # embed tokens and positions
        x, _ = self.forward_embedding(
            prev_output_tokens, token_embeddings, incremental_state
        )
        is_first_step = self.is_first_step(incremental_state)

        # relative position
        self_attn_rel_pos_bias = None
        slen = prev_output_tokens.size(1)

        if self.self_attn_relative_position is not None:
            self_attn_rel_pos_bias = self.self_attn_relative_position(
                batch_size=x.size(0), qlen=slen, klen=slen
            )
            if incremental_state is not None and not is_first_step:
                self_attn_rel_pos_bias = self_attn_rel_pos_bias[-1:, :, :]
        cross_attn_rel_pos_bias = None

        if self.cross_attn_relative_position is not None:
            cross_attn_rel_pos_bias = self.cross_attn_relative_position(
                batch_size=x.size(0),
                qlen=slen,
                klen=encoder_out["encoder_out"].size(1),
            )
            if incremental_state is not None and not is_first_step:
                cross_attn_rel_pos_bias = cross_attn_rel_pos_bias[-1:, :, :]

        # decoder layers
        inner_states = [x]

        if encoder_out is None:
            l_aux = []
        else:
            l_aux = encoder_out["l_aux"] if "l_aux" in encoder_out else []

        # Process each token through the stacked decoder layers.
        for idx, layer in enumerate(self.layers):
            if incremental_state is None or is_first_step:
                self_attn_mask = torch.triu(
                    torch.zeros([x.size(1), x.size(1)])
                    .float()
                    .fill_(float("-inf"))
                    .type_as(x),
                    1,
                )
                if is_first_step and incremental_state is not None:
                    if idx not in incremental_state:
                        incremental_state[idx] = {}
            else:
                self_attn_mask = None
                if idx not in incremental_state:
                    incremental_state[idx] = {}

            x, layer_attn, _, l_aux_i = layer(
                x,
                encoder_out["encoder_out"] if encoder_out is not None else None,
                encoder_out["encoder_padding_mask"]
                if encoder_out is not None
                else None,
                incremental_state[idx] if incremental_state is not None else None,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                self_attn_rel_pos=self_attn_rel_pos_bias,
                cross_attn_rel_pos=cross_attn_rel_pos_bias,
                is_first_step=is_first_step,
            )
            l_aux.append(l_aux_i)
            inner_states.append(x)

        # Apply the final layer normalization if specified.
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # Output projection layer converts the decoder's output to token probabilities.
        if not features_only:
            x = self.output_layer(x)

        return x, {
            "inner_states": inner_states,
            "l_aux": l_aux,
            "attn": None,
        }

    # The output_layer method applies the final linear transformation to the features.
    def output_layer(self, features):
        return self.output_projection(features)
