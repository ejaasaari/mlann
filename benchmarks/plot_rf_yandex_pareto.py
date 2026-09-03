#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE_RESULTS = ROOT / "rf_yandex_threshold_baseline.csv"
DEFAULT_OUTPUT = ROOT / "rf_yandex_pareto.png"
TARGET_RECALL_MIN = 0.70
TARGET_RECALL_MAX = 0.95


@dataclass(frozen=True)
class Series:
    label: str
    path: Path


def parse_series(value):
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=CSV")
    return Series(label.strip(), Path(path).expanduser().resolve())


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot one or more RF result sets as global Pareto frontiers."
    )
    parser.add_argument(
        "--series",
        action="append",
        type=parse_series,
        metavar="LABEL=CSV",
        help="result set to compare with the baseline; repeat for more CSV files",
    )
    parser.add_argument(
        "--baseline", type=Path, default=DEFAULT_BASELINE_RESULTS, help="baseline result CSV"
    )
    parser.add_argument("--baseline-label", default="Threshold baseline")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No measurements in {path}")
    for row in rows:
        for key in (
            "trees",
            "depth",
            "leaf_votes",
            "queries",
            "warmup_queries",
            "repetitions",
        ):
            row[key] = int(row[key])
        for key in ("recall_at_100", "query_ms", "build_time_s"):
            row[key] = float(row[key])
    return rows


def pareto_frontier(rows):
    frontier = []
    best_recall = float("-inf")
    for row in sorted(rows, key=lambda value: (value["query_ms"], -value["recall_at_100"])):
        if row["recall_at_100"] > best_recall:
            frontier.append(row)
            best_recall = row["recall_at_100"]
    return frontier


def main():
    arguments = parse_arguments()
    series = [
        Series(arguments.baseline_label, arguments.baseline.expanduser().resolve()),
        *(arguments.series or []),
    ]
    loaded = [(item, load_rows(item.path)) for item in series]
    frontiers = []
    for item, rows in loaded:
        frontier = [
            row
            for row in pareto_frontier(rows)
            if TARGET_RECALL_MIN <= row["recall_at_100"] <= TARGET_RECALL_MAX
        ]
        if not frontier:
            raise ValueError(
                f"No global Pareto-frontier measurements from {item.path} fall in recall range "
                f"[{TARGET_RECALL_MIN}, {TARGET_RECALL_MAX}]"
            )
        frontier.sort(key=lambda value: value["recall_at_100"])
        frontiers.append((item, frontier))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    all_rows = [row for _, rows in loaded for row in rows]
    query_counts = sorted({row["queries"] for row in all_rows})
    warmup_counts = sorted({row["warmup_queries"] for row in all_rows})
    repetitions = sorted({row["repetitions"] for row in all_rows})
    configuration_sets = [
        {(row["trees"], row["depth"], row["leaf_votes"]) for row in rows}
        for _, rows in loaded
    ]
    shared_configurations = set.intersection(*configuration_sets)
    max_trees = max(row["trees"] for row in all_rows)

    def values_label(values):
        return "/".join(f"{value:,}" for value in values)

    repetition_label = (
        "one timed pass"
        if repetitions == [1]
        else f"median of {values_label(repetitions)} timed passes"
    )
    configuration_label = f"{len(shared_configurations)} (trees, depth, leaf-votes) configurations"
    if len(frontiers) > 1:
        configuration_label = "shared " + configuration_label

    styles = (
        {"color": "#111111", "linestyle": "-", "marker": "o"},
        {"color": "#d95f02", "linestyle": "--", "marker": "s"},
        {"color": "#1b9e77", "linestyle": "-.", "marker": "^"},
        {"color": "#7570b3", "linestyle": ":", "marker": "D"},
    )
    for index, (item, frontier) in enumerate(frontiers):
        style = styles[index % len(styles)]
        ax.plot(
            [value["recall_at_100"] for value in frontier],
            [value["query_ms"] for value in frontier],
            linewidth=2.2,
            markersize=6,
            label=item.label,
            zorder=10 - index,
            **style,
        )
    title = "RF on yandex-200-cosine: global Pareto frontier"
    if len(frontiers) > 1:
        title += " comparison"
    ax.set_title(title, pad=12)
    ax.set_xlabel("Recall@100 (higher is better)")
    ax.set_ylabel("Query latency (ms; log scale, lower is better)")
    ax.set_xlim(0.695, 0.955)
    ax.set_yscale("log")
    frontier_rows = [row for _, frontier in frontiers for row in frontier]
    min_latency = min(row["query_ms"] for row in frontier_rows)
    max_latency = max(row["query_ms"] for row in frontier_rows)
    ax.set_ylim(min_latency / 1.15, max_latency * 1.15)
    latency_ticks = [0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100]
    ax.yaxis.set_major_locator(
        FixedLocator(
            [
                tick
                for tick in latency_ticks
                if min_latency / 1.15 <= tick <= max_latency * 1.15
            ]
        )
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(which="minor", axis="y", alpha=0.18)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.text(
        0.01,
        0.99,
        f"Only each result set's global nondominated points with "
        f"{TARGET_RECALL_MIN:.2f} ≤ recall ≤ {TARGET_RECALL_MAX:.2f} are shown\n"
        f"{configuration_label} · maximum {max_trees} trees\n"
        f"{values_label(query_counts)} queries · {values_label(warmup_counts)} warmups · "
        f"{repetition_label} · single-thread latency",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#333333",
    )
    fig.savefig(arguments.output.resolve(), dpi=180)


if __name__ == "__main__":
    main()
