"""CSV lookup helpers shared by the experiment SRAM-CIM configurations."""

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EFFICIENCY_TABLE = REPO_ROOT / "experiments" / "docs" / "SRAM_DATA" / "cim_macro_efficiency.csv"


def lookup_cim_macro_efficiency(
    accumulation_length: int,
    parallel_columns: int,
    weight_bits: int,
    input_bits: int,
    metric: str,
) -> float:
    """Return a characterization-table value for one exact macro configuration."""

    with EFFICIENCY_TABLE.open(newline="") as table:
        for row in csv.DictReader(table):
            if (
                int(row["AL"]) == accumulation_length
                and int(row["PC"]) == parallel_columns
                and int(row["WB"]) == weight_bits
                and int(row["IB"]) == input_bits
            ):
                return float(row[metric])

    raise ValueError(
        "No SRAM-CIM characterization entry for "
        f"AL={accumulation_length}, PC={parallel_columns}, "
        f"WB={weight_bits}, IB={input_bits}."
    )
