#!/usr/bin/env python3
"""Profile one Tile/layer scheduling search with deterministic diagnostics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


RUN_DIR = Path(__file__).resolve().parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from sram_macro_types import load_sram_macro_library
from tile_DSE.sram_tile_dse import (
    LayerShape,
    ScheduleSearchMetrics,
    enumerate_tile_designs,
    evaluate_tile_schedules,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", default="5xm32x8+4xm128x8")
    parser.add_argument("--reduction-length", type=int, default=4608)
    parser.add_argument("--output-channels", type=int, default=512)
    parser.add_argument("--output-positions", type=int, default=49)
    parser.add_argument("--latency-cap-ratio", type=float, default=4.0)
    parser.add_argument("--local-pareto-epsilon", type=float, default=0.001)
    parser.add_argument("--local-state-cap", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = load_sram_macro_library("system_digital_sram_4x4")
    reference, designs = enumerate_tile_designs(library, 0.4608, 0.20)
    design_by_name = {design.name: design for design in designs}
    if args.design not in design_by_name:
        raise ValueError(f"Unknown Tile design: {args.design}")

    layer = LayerShape(
        "profile",
        args.reduction_length,
        args.output_channels,
        args.output_positions,
    )
    metrics = ScheduleSearchMetrics()
    rows = evaluate_tile_schedules(
        layer,
        design_by_name[args.design],
        library,
        library.tech_node,
        reference,
        args.latency_cap_ratio,
        args.local_pareto_epsilon,
        args.local_state_cap,
        metrics=metrics,
    )
    signature = repr(
        sorted(
            (
                row["fabric_energy_j"],
                row["latency_s"],
                row["schedule_signature"],
            )
            for row in rows
        )
    ).encode()
    result = {
        "design": args.design,
        "reduction_length": args.reduction_length,
        "output_channels": args.output_channels,
        "output_positions": args.output_positions,
        "pareto_rows": len(rows),
        "result_sha256": hashlib.sha256(signature).hexdigest(),
        **asdict(metrics),
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            writer.writeheader()
            writer.writerow(result)


if __name__ == "__main__":
    main()
