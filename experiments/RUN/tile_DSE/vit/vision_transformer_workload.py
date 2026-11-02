"""Corrected ViT workloads used by the ViT Tile DSE."""

from __future__ import annotations

from dataclasses import dataclass

from tile_DSE.sram_tile_dse import LayerShape


@dataclass(frozen=True)
class VitOperatorGroup:
    """Repeated operators that share one P/K/N shape and mapping role."""

    name: str
    stream: str
    shape: LayerShape
    repeat: int

    @property
    def macs(self) -> int:
        return self.shape.macs * self.repeat


def _vit_operator_groups(
    *,
    batch_size: int,
    tokens: int,
    heads: int,
    hidden_dimension: int,
    mlp_dimension: int,
    blocks: int,
) -> tuple[VitOperatorGroup, ...]:
    """Build a ViT workload with explicit sequence and hidden dimensions.

    Attention folds ``batch * heads * query_tokens`` into output_positions.
    This preserves all 192 query tokens that are absent from the generic
    two-dimensional attention representation in the upstream example YAML.
    """

    if batch_size < 1 or tokens < 1 or heads < 1 or hidden_dimension < 1:
        raise ValueError("workload dimensions and batch_size must be positive")
    if hidden_dimension % heads:
        raise ValueError("hidden_dimension must be divisible by heads")

    head_dimension = hidden_dimension // heads
    attention_positions = batch_size * heads * tokens

    return (
        VitOperatorGroup(
            "patch_embedding",
            "endpoint",
            LayerShape("patch_embedding", 16 * 16 * 3, hidden_dimension, batch_size * tokens),
            1,
        ),
        VitOperatorGroup(
            "q_k_v_o_projection",
            "projection",
            LayerShape("q_k_v_o_projection", hidden_dimension, hidden_dimension, batch_size * tokens),
            blocks * 4,
        ),
        VitOperatorGroup(
            "ffn_up_projection",
            "projection",
            LayerShape("ffn_up_projection", hidden_dimension, mlp_dimension, batch_size * tokens),
            blocks,
        ),
        VitOperatorGroup(
            "ffn_down_projection",
            "projection",
            LayerShape("ffn_down_projection", mlp_dimension, hidden_dimension, batch_size * tokens),
            blocks,
        ),
        VitOperatorGroup(
            "attention_qk",
            "attention",
            LayerShape("attention_qk", head_dimension, tokens, attention_positions),
            blocks,
        ),
        VitOperatorGroup(
            "attention_av",
            "attention",
            LayerShape("attention_av", tokens, head_dimension, attention_positions),
            blocks,
        ),
        VitOperatorGroup(
            "classifier",
            "endpoint",
            LayerShape("classifier", hidden_dimension, 1000, batch_size),
            1,
        ),
    )


def vit_b16_operator_groups(batch_size: int = 1) -> tuple[VitOperatorGroup, ...]:
    """Return the corrected ViT-B/16 workload (192 tokens, 768 hidden)."""

    return _vit_operator_groups(
        batch_size=batch_size,
        tokens=192,
        heads=12,
        hidden_dimension=768,
        mlp_dimension=3072,
        blocks=12,
    )


def vit_l16_operator_groups(batch_size: int = 1) -> tuple[VitOperatorGroup, ...]:
    """Return ViT-L/16 with 192 tokens and 1024 hidden dimension."""

    return _vit_operator_groups(
        batch_size=batch_size,
        tokens=192,
        heads=16,
        hidden_dimension=1024,
        mlp_dimension=4096,
        blocks=24,
    )


def vit_b16_macs(batch_size: int = 1) -> int:
    return sum(group.macs for group in vit_b16_operator_groups(batch_size))


def vit_l16_macs(batch_size: int = 1) -> int:
    return sum(group.macs for group in vit_l16_operator_groups(batch_size))
