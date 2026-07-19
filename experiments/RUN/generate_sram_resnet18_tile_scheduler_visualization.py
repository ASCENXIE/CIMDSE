"""Plot per-layer effective energy efficiency for the ResNet-18 Tile DSE."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/accelforge_mplconfig")

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results" / "sram_resnet18_tile_scheduler_results.csv"
FIGURE = ROOT / "experiments" / "results" / "sram_resnet18_tile_scheduler_efficiency.png"
DESIGNS = ("nine_m64x8", "m64x8_m512x8")
LABELS = {"nine_m64x8": "9 x 64x8 Tile", "m64x8_m512x8": "64x8 + 512x8 Tile"}
COLORS = {"nine_m64x8": "#c75b39", "m64x8_m512x8": "#2e86ab"}


def _layer_index(layer: str) -> int:
    return int(layer.removeprefix("T"))


def read_efficiency() -> tuple[list[str], dict[str, list[float]]]:
    by_layer: dict[str, dict[str, float]] = defaultdict(dict)
    with RESULTS.open(newline="") as file:
        for row in csv.DictReader(file):
            by_layer[row["layer"]][row["design"]] = float(row["TOPS_per_W"])
    layers = sorted(by_layer, key=_layer_index)
    if any(design not in by_layer[layer] for layer in layers for design in DESIGNS):
        raise ValueError("Every ResNet-18 layer must have both Tile designs.")
    return layers, {design: [by_layer[layer][design] for layer in layers] for design in DESIGNS}


def main() -> None:
    layers, values = read_efficiency()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(layers))
    width = 0.36
    figure, axis = plt.subplots(figsize=(18, 6.4))
    for offset, design in ((-width / 2, DESIGNS[0]), (width / 2, DESIGNS[1])):
        axis.bar(x + offset, values[design], width=width, color=COLORS[design], label=LABELS[design])
    axis.set_title("ResNet-18 Per-Layer Effective Energy Efficiency", pad=14)
    axis.set_ylabel("effective energy efficiency (TOPS/W)")
    axis.set_xlabel("ResNet-18 layer")
    axis.set_xticks(x)
    axis.set_xticklabels(layers)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=False)
    figure.text(
        0.5,
        0.01,
        "Weight-stationary Tile DSE: Macro VMM + vector psum energy only.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print(FIGURE)


if __name__ == "__main__":
    main()
