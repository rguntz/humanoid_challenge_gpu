"""Transformer Q-critic for chunked actions - a Flax reimplementation of
srl_il_dev's QChunkingTransformer/MIMO_Transformer
(srl_il/algo/q_chunking.py, srl_il/models/common/transformer.py), matched
to that reference exactly EXCEPT: the HL-Gauss value-support bounds/sigma
(task-specific in the reference, re-derived for our own reward scale - see
critic.py) and the observation history length (To=2 here, matching
xqc_style/mlp/MIP/MF/FM's own convention, vs the reference's own state
experiment's To=1 - see conversation history for both).

Reference: https://github.com/srl-ethz/srl_il_dev
(commit 2e60556ec4a45717ae59e45d2dac263363b1eaf1, local checkout at
~/workspace/srl_il_dev), specifically:
  - srl_il/algo/q_chunking.py (QChunkingTransformer, HLGaussLoss)
  - srl_il/models/common/transformer.py (MIMO_Transformer, TransformerEncoderLayer)
  - srl_il/models/common/position_encoding.py (PositionEmbeddingSineSeq)
  - srl_il/models/common/misc.py (EnsembleModule)
  - srl_il/algo/mixin_obs.py (LowdimConcat, obs tokenization/posemb wiring)
  - srl_il/cfg/adroit/rl_offline_actqc.yaml (the state-experiment config
    this was checked against for To=1, lowdim_concat, whole_seq_sine)

Tokenization: To obs tokens (one per timestep, Dense(d_model), a FIXED
sinusoidal position embedding - matches PositionEmbeddingSineSeq exactly)
concatenated with h action tokens (one per chunk step, elu(Dense(d_model)),
a SEPARATELY-LEARNED per-slot position bias - matches
QChunkingTransformer's own action_pos_bias, a genuinely different scheme
from the obs tokens' sinusoidal one, not a shared/unified position index -
see conversation history for why this dual-scheme choice is deliberate),
plus one learned CLS query token (zero position, matching MIMO_Transformer's
own `torch.zeros_like(cls_embed)`). A standard POST-LN bidirectional
transformer encoder (matches MIMO_Transformer's own normalize_before=False
default, unchanged by every reference config that uses it - DETR-style
blocks: positional embedding added to query/key only, never to value) reads
the CLS token's final hidden state through Dense(d_model, num_bins) to
produce the HL-Gauss logits (see critic.py).

Ensembling: VMapTransformerCritic wraps TransformerCritic in nn.vmap over
n_critics independent parameter sets - the same trick
xqc/networks/critic_net.py's VMapCritic already uses for xqc_style/mlp,
functionally identical to the reference's own EnsembleModule (confirmed:
just num_heads independent nn.Module copies, srl_il/models/common/misc.py).
"""
import math

import flax.linen as nn
import jax.numpy as jnp
import numpy as np


def sine_position_embedding(length: int, d_model: int, temperature: float = 10000.0) -> jnp.ndarray:
    """(length, d_model) fixed sinusoidal position embedding - matches
    PositionEmbeddingSineSeq.positional_encoding exactly: even feature
    dims get sin, odd dims get cos, same div_term formula. A pure function
    of static shapes, so this is just a constant baked in at trace time -
    no trainable parameters."""
    position = np.arange(length, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(temperature) / d_model))
    pe = np.zeros((length, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return jnp.asarray(pe)


class _MultiHeadSelfAttention(nn.Module):
    """Manual multi-head self-attention (rather than
    flax.linen.MultiHeadDotProductAttention) so query/key can take a
    different input (src+pos) than value (src) - matches
    TransformerEncoderLayer.forward_post's `with_pos_embed` exactly:
    `q = k = src + pos; attn(q, k, value=src)`, positional embedding never
    added to value."""
    d_model: int
    nhead: int

    @nn.compact
    def __call__(self, qk_input: jnp.ndarray, v_input: jnp.ndarray) -> jnp.ndarray:
        head_dim = self.d_model // self.nhead
        q = nn.Dense(self.d_model, name="q_proj")(qk_input)
        k = nn.Dense(self.d_model, name="k_proj")(qk_input)
        v = nn.Dense(self.d_model, name="v_proj")(v_input)

        def split_heads(x):
            b, t, _ = x.shape
            return x.reshape(b, t, self.nhead, head_dim)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn_logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(head_dim)
        attn_weights = nn.softmax(attn_logits, axis=-1)
        out = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        out = out.reshape(out.shape[0], out.shape[1], self.d_model)
        return nn.Dense(self.d_model, name="out_proj")(out)


class TransformerEncoderLayer(nn.Module):
    """Post-LN encoder block - matches TransformerEncoderLayer.forward_post
    exactly (normalize_before=False is the reference's own default,
    unchanged by any config that uses it): self-attn -> dropout -> residual
    -> LayerNorm -> FFN (GELU, matching critic_network_cfg's own
    activation='gelu') -> dropout -> residual -> LayerNorm."""
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float

    @nn.compact
    def __call__(self, src: jnp.ndarray, pos: jnp.ndarray, training: bool) -> jnp.ndarray:
        qk = src + pos
        attn_out = _MultiHeadSelfAttention(self.d_model, self.nhead)(qk, src)
        src = src + nn.Dropout(self.dropout, deterministic=not training)(attn_out)
        src = nn.LayerNorm()(src)

        ff = nn.Dense(self.dim_feedforward)(src)
        ff = nn.gelu(ff)
        ff = nn.Dropout(self.dropout, deterministic=not training)(ff)
        ff = nn.Dense(self.d_model)(ff)
        src = src + nn.Dropout(self.dropout, deterministic=not training)(ff)
        src = nn.LayerNorm()(src)
        return src


class TransformerCritic(nn.Module):
    """One critic instance - see module docstring for the tokenization/
    readout scheme. `To`/`h` are fixed by the chunked buffer, not learned."""
    To: int
    h: int
    d_model: int
    nhead: int
    num_encoder_layers: int
    dim_feedforward: int
    dropout: float
    num_bins: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool) -> jnp.ndarray:
        batch_size = observations.shape[0]

        obs_tok = nn.Dense(self.d_model, name="obs_proj")(observations)  # (B, To, d_model)
        obs_pos = sine_position_embedding(self.To, self.d_model)[None]  # (1, To, d_model)

        act_tok = nn.elu(nn.Dense(self.d_model, name="action_proj")(actions))  # (B, h, d_model)
        action_pos_bias = self.param(
            "action_pos_bias", nn.initializers.xavier_uniform(), (1, self.h, self.d_model)
        )

        cls_token = self.param(
            "cls_token", nn.initializers.normal(stddev=0.02), (1, 1, self.d_model)
        )
        cls_tok = jnp.broadcast_to(cls_token, (batch_size, 1, self.d_model))

        tokens = jnp.concatenate([obs_tok, act_tok, cls_tok], axis=1)
        pos = jnp.concatenate(
            [
                jnp.broadcast_to(obs_pos, (batch_size, self.To, self.d_model)),
                jnp.broadcast_to(action_pos_bias, (batch_size, self.h, self.d_model)),
                jnp.zeros((batch_size, 1, self.d_model)),
            ],
            axis=1,
        )

        hs = tokens
        for _ in range(self.num_encoder_layers):
            hs = TransformerEncoderLayer(
                self.d_model, self.nhead, self.dim_feedforward, self.dropout
            )(hs, pos, training)

        cls_out = hs[:, -1, :]
        return nn.Dense(self.num_bins, name="q_head")(cls_out)


class VMapTransformerCritic(nn.Module):
    """n_critics independent TransformerCritic copies - same nn.vmap-over-
    params trick as xqc/networks/critic_net.py's VMapCritic, functionally
    identical to the reference's own EnsembleModule."""
    To: int
    h: int
    d_model: int
    nhead: int
    num_encoder_layers: int
    dim_feedforward: int
    dropout: float
    num_bins: int
    n_critics: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool) -> jnp.ndarray:
        logits = nn.vmap(
            TransformerCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )(
            self.To,
            self.h,
            self.d_model,
            self.nhead,
            self.num_encoder_layers,
            self.dim_feedforward,
            self.dropout,
            self.num_bins,
        )(observations, actions, training)
        return logits  # (n_critics, B, num_bins)
