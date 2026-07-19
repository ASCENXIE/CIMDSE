"""Workload-shape helpers shared by SRAM-CIM design-space exploration scripts."""

from __future__ import annotations

import re


SIMPLE_BOUND = re.compile(r"^0\s*<=\s*([A-Za-z_]\w*)\s*<\s*(\d+)$")


def rank_variable_sizes(einsum, workload) -> dict[str, int]:
    sizes = {}
    for constraint in einsum.iteration_space_shape:
        match = SIMPLE_BOUND.match(str(constraint).replace(" ", ""))
        if match:
            sizes[match.group(1)] = int(match.group(2))
    for rank_var in set(einsum.rank_variables) - set(sizes):
        candidates = {
            int(size)
            for rank in einsum.rank_variable2ranks[rank_var]
            for size in (einsum.rank_sizes.get(rank), workload.rank_sizes.get(rank))
            if size is not None
        }
        if len(candidates) == 1:
            sizes[rank_var] = candidates.pop()
    missing = set(einsum.rank_variables) - set(sizes)
    if missing:
        raise ValueError(f"Missing rank sizes for {einsum.name}: {sorted(missing)}")
    return sizes


def tensor_for_role(einsum, role: str):
    source = next(rename.source for rename in einsum.renames if rename.name == role)
    if isinstance(source, str):
        return next(tensor for tensor in einsum.tensor_accesses if tensor.name == source)
    return next(tensor for tensor in einsum.tensor_accesses if tensor.name in source)
