"""Plot the original first-round Tile GP fit and a slice of its predictions."""

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experiments/results/tile_DSE/v5"
CANDIDATES = RESULTS / "preparation_250mhz_bus4_development_20261117/candidates_134.csv"
STATE = RESULTS / "e04_development48_250mhz_bus4_20261117/state.json"
REFERENCE = "2xm256x8"
FIRST_PICK = "1xm128x8+2xm256x8"
ROW32_BRANCH = tuple(f"{count}xm32x8+{REFERENCE}" for count in range(1, 6))

sys.path.insert(0, str(ROOT / "experiments/RUN"))
from tile_DSE.v5.campaign import DEVELOPMENT, measured_development
from tile_DSE.v5.joint_bo import MACRO_NAMES, _inputs, _model


def read_candidates():
    with CANDIDATES.open(encoding="utf-8", newline="") as source:
        designs = []
        for row in csv.DictReader(source):
            counts = []
            for component in row["design"].split("+"):
                count, macro = component.split("x", 1)
                counts.append((macro, int(count)))
            designs.append(SimpleNamespace(name=row["design"], counts=tuple(counts)))
        return designs


def tile_label(design):
    return "+".join(
        f"{count}×{macro[1:-2]}" if count > 1 else macro[1:-2]
        for macro, count in design.counts
    )


def training_table(designs, features, index, scores):
    names = sorted(scores, key=lambda name: scores[name]["geometric_mean"])
    coordinates = features[[index[name] for name in names]]
    score_values = [f"{scores[name]['geometric_mean']:.3f}" for name in names]
    return go.Table(
        columnwidth=[2, 1, 1, 1, 1, 1, 1.4],
        header=dict(
            values=["Tile", *(f"{macro[1:-2]} 行" for macro in MACRO_NAMES), "回放分数 G"],
            fill_color="#19344d", font=dict(color="white"), align="center", height=30,
        ),
        cells=dict(
            values=[
                [tile_label(designs[index[name]]) for name in names],
                *[[f"{value:.2g}" for value in column] for column in coordinates.T],
                score_values,
            ],
            fill_color=[
                ["#e9f2fb" if name == REFERENCE else "#f3f6fa" for name in names],
                *[["#dbeafb" if value else "#f8fafc" for value in column] for column in coordinates.T],
                ["#d3eee5" if name == REFERENCE else "#eff8f5" for name in names],
            ],
            line_color="white", align="center", height=24,
        ),
    )


def make_figure():
    state = json.loads(STATE.read_text(encoding="utf-8"))
    initial_names = set(state["selection"][0]["designs"])
    initial = measured_development({
        "completed": {name: rows for name, rows in state["completed"].items() if name in initial_names},
    })
    designs, features, index, measured, scores = _inputs(read_candidates(), initial)
    train_x = features[[index[name] for name in measured]]
    train_y = np.log([scores[name]["geometric_mean"] for name in measured])
    reference = features[index[REFERENCE]]
    baseline_score = scores[REFERENCE]["geometric_mean"]

    figure = make_subplots(
        rows=2, cols=1, specs=[[{"type": "table"}], [{"type": "xy"}]],
        row_heights=[0.53, 0.47], vertical_spacing=0.12,
        subplot_titles=(
            f"1. 首批训练数据（{len(measured)} 个双网样本）",
            "2. 同一个起点，两种增加 Macro 的方式",
        ),
    )
    figure.add_trace(training_table(designs, features, index, scores), row=1, col=1)
    figure.add_trace(go.Scatter(
        x=[0, 0.25], y=[baseline_score, baseline_score], mode="lines",
        line=dict(color="#7890a7", dash="dot", width=1.4), hoverinfo="skip", showlegend=False,
    ), row=2, col=1)

    for branch, name, color, symbol in (
        (ROW32_BRANCH, "预测 · 增加 32 行 Macro", "#126bb0", "circle"),
        ((FIRST_PICK,), "预测 · 增加 128 行 Macro", "#823e9c", "diamond"),
    ):
        coordinates = features[[index[tile] for tile in branch]]
        distance = np.linalg.norm(coordinates - reference, axis=1)
        mean, deviation = _model(train_x, train_y, coordinates)
        center, lower, upper = np.exp(mean), np.exp(mean - deviation), np.exp(mean + deviation)
        labels = [f"{value:.3f}" if np.isclose(offset, 0.2) else ""
                  for offset, value in zip(distance, center)]
        details = [
            [tile_label(designs[index[tile]]), str([float(value) for value in point]), low, high]
            for tile, point, low, high in zip(branch, coordinates, lower, upper)
        ]
        figure.add_trace(go.Scatter(
            name=name, x=distance, y=center, mode="markers+text",
            text=labels, textposition="middle left", textfont=dict(color=color),
            marker=dict(color=color, symbol=symbol, size=11),
            error_y=dict(type="data", symmetric=False, array=upper - center,
                         arrayminus=center - lower, color=color, thickness=1.5),
            customdata=details,
            hovertemplate=("<b>%{customdata[0]}</b><br>数量坐标 %{customdata[1]}"
                           "<br>距离 %{x:.2f}<br>预测 G %{y:.3f}"
                           "<br>预测区间（ln(G) ±1σ） %{customdata[2]:.3f}–%{customdata[3]:.3f}"
                           "<extra></extra>"),
        ), row=2, col=1)

    first_pick = measured_development(state)[FIRST_PICK]
    initial_minima = {
        workload: min(score["best_schedules"][workload]["edap_j_s_mm2"] for score in scores.values())
        for workload in DEVELOPMENT
    }
    observed = np.sqrt(np.prod([
        first_pick[workload][0]["edap_j_s_mm2"] / initial_minima[workload] for workload in DEVELOPMENT
    ]))
    picked_distance = float(np.linalg.norm(features[index[FIRST_PICK]] - reference))
    for distance, score, label, color, symbol in (
        (0, baseline_score, "首批回放 · 2×256", "#007b63", "square"),
        (picked_distance, observed, "随后回放 · 128+2×256", "#b54d11", "x"),
    ):
        figure.add_trace(go.Scatter(
            name=label, x=[distance], y=[score], mode="markers",
            marker=dict(color=color, symbol=symbol, size=12),
            hovertemplate=f"<b>{label}</b><br>距离 %{{x:.2f}}<br>回放 G %{{y:.3f}}<extra></extra>",
        ), row=2, col=1)

    figure.update_xaxes(
        title_text="与 2×256 的五维数量坐标距离", range=[-0.02, 0.29], dtick=0.05,
        tickformat=".2f", row=2, col=1,
    )
    figure.update_yaxes(title_text="双网分数 G（越低越好）", row=2, col=1)
    figure.update_layout(
        title=dict(text="从 Macro 数量坐标到双网分数", x=0.05),
        height=970, template="plotly_white",
        font=dict(family="Noto Sans CJK SC, Microsoft YaHei, sans-serif", color="#19344d", size=14),
        margin=dict(l=65, r=35, t=100, b=110), legend=dict(orientation="h", y=-0.11),
    )
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "outer_tile_selection_gp.html")
    args = parser.parse_args()
    figure = make_figure()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(args.output, include_plotlyjs=True, config={"displaylogo": False, "responsive": True})
    print(args.output)


if __name__ == "__main__":
    main()
