#!/usr/bin/env python3
"""Benchmark the current threshold-based RF query on Yandex."""

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
SOURCE = BENCHMARK_DIR / "mlann_example.cpp"
BINARY = BENCHMARK_DIR / ".build" / "mlann_example"
BASELINE_RESULTS = BENCHMARK_DIR / "rf_yandex_threshold_baseline.csv"
DEFAULT_RESULTS = BENCHMARK_DIR / "rf_yandex_results.csv"
PLOTTER = BENCHMARK_DIR / "plot_rf_yandex_pareto.py"
FIELDS = (
    "config",
    "trees",
    "depth",
    "leaf_votes",
    "votes_required",
    "average_candidates",
    "recall_at_100",
    "query_ms",
    "build_time_s",
    "queries",
    "warmup_queries",
    "repetitions",
)

BUILD_PATTERN = re.compile(r"Build time \(s\): (?P<seconds>[0-9.]+)")
NUMBER = r"[0-9.eE+-]+"
RESULT_PATTERN = re.compile(
    rf"votes_required=(?P<votes_required>{NUMBER}) "
    rf"recall=(?P<recall>{NUMBER}) "
    rf"query_ms=(?P<query_ms>{NUMBER}) "
    rf"elected=(?P<elected>{NUMBER})"
)


@dataclass(frozen=True)
class Configuration:
    trees: int
    depth: int
    leaf_votes: int

    @property
    def name(self):
        return f"T{self.trees}-D{self.depth}-B{self.leaf_votes}"


CONFIGURATIONS = (
    Configuration(60, 14, 1),
    Configuration(60, 15, 1),
    Configuration(60, 16, 1),
    Configuration(20, 14, 1),
    Configuration(20, 15, 1),
    Configuration(20, 16, 1),
)
DEFAULT_VOTES_REQUIRED = (
    0.0000005,
    0.000001,
    0.0000015,
    0.000002,
    0.000003,
    0.000004,
    0.000005,
    0.000007,
    0.000010,
    0.000015,
    0.000020,
)


def comma_separated_floats(value):
    try:
        parsed = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("votes_required thresholds must be positive")
    return parsed


def pkg_config(*arguments):
    try:
        output = subprocess.check_output(
            ["pkg-config", *arguments, "hdf5"], text=True, stderr=subprocess.STDOUT
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("pkg-config could not locate the HDF5 development package") from error
    return shlex.split(output)


def compile_benchmark(cxx):
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    command = [
        *shlex.split(cxx),
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-fopenmp",
        "-march=native",
        f"-I{REPO_ROOT / 'cpp' / 'lib'}",
        *pkg_config("--cflags"),
        str(SOURCE),
        "-lhdf5_cpp",
        *pkg_config("--libs"),
        "-o",
        str(BINARY),
    ]
    print("Compiling benchmark:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def write_results(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_configuration(configuration, arguments):
    dataset_name = arguments.dataset.stem
    votes_required = ",".join(f"{value:.9g}" for value in arguments.votes_required)
    command = [
        str(BINARY),
        dataset_name,
        f"--trees={configuration.trees}",
        f"--depth={configuration.depth}",
        f"--leaf-votes={configuration.leaf_votes}",
        f"--votes-required={votes_required}",
        f"--queries={arguments.queries}",
        f"--warmup={arguments.warmup}",
        f"--query-repeats={arguments.query_repeats}",
    ]
    print(f"\nRunning {configuration.name}:", shlex.join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=arguments.dataset.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    build_time = None
    measurements = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if match := BUILD_PATTERN.search(line):
            build_time = float(match.group("seconds"))
        if match := RESULT_PATTERN.search(line):
            measurements.append(
                {
                    "config": configuration.name,
                    "trees": configuration.trees,
                    "depth": configuration.depth,
                    "leaf_votes": configuration.leaf_votes,
                    "votes_required": match.group("votes_required"),
                    "average_candidates": match.group("elected"),
                    "recall_at_100": match.group("recall"),
                    "query_ms": match.group("query_ms"),
                    "build_time_s": build_time,
                    "queries": arguments.queries,
                    "warmup_queries": arguments.warmup,
                    "repetitions": arguments.query_repeats,
                }
            )

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if build_time is None or len(measurements) != len(arguments.votes_required):
        raise RuntimeError(f"Incomplete benchmark output for {configuration.name}")
    return measurements


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "yandex-200-cosine.hdf5",
        help="path to yandex-200-cosine.hdf5",
    )
    parser.add_argument(
        "--votes-required",
        type=comma_separated_floats,
        default=DEFAULT_VOTES_REQUIRED,
        help="comma-separated RF probability thresholds",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS,
        help="result CSV; defaults to a non-baseline file so the baseline is preserved",
    )
    parser.add_argument("--label", default="Current RF", help="label used in the comparison plot")
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--query-repeats", type=int, default=1)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    arguments = parser.parse_args()
    arguments.dataset = arguments.dataset.resolve()
    arguments.output = arguments.output.resolve()
    if not arguments.dataset.is_file():
        parser.error(f"dataset does not exist: {arguments.dataset}")
    if arguments.dataset.stem != "yandex-200-cosine":
        parser.error("the benchmark driver expects a file named yandex-200-cosine.hdf5")
    if arguments.queries <= 0 or arguments.warmup < 0 or arguments.query_repeats <= 0:
        parser.error("queries and query-repeats must be positive; warmup must be non-negative")
    return arguments


def main():
    arguments = parse_arguments()
    if not arguments.skip_build:
        compile_benchmark(arguments.cxx)
    elif not BINARY.is_file():
        raise FileNotFoundError(f"benchmark binary does not exist: {BINARY}")

    rows = []
    for configuration in CONFIGURATIONS:
        rows.extend(run_configuration(configuration, arguments))
        # Preserve every completed configuration if this long benchmark is interrupted.
        write_results(arguments.output, FIELDS, rows)

    if not arguments.no_plot:
        plot_command = [sys.executable, str(PLOTTER)]
        if arguments.output != BASELINE_RESULTS.resolve():
            plot_command.extend(["--series", f"{arguments.label}={arguments.output}"])
        subprocess.run(plot_command, cwd=BENCHMARK_DIR, check=True)
        print(f"\nPlot: {BENCHMARK_DIR / 'rf_yandex_pareto.png'}")
    print(f"Results: {arguments.output}")


if __name__ == "__main__":
    main()
