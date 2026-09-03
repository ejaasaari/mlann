# RF Yandex Pareto benchmark

Benchmark the current RF implementation and compare it with the preserved threshold baseline:

```bash
python3 benchmarks/run_rf_yandex_pareto.py
```

The runner uses RF's native `votes_required` probability-threshold query. It writes new runs to
`benchmarks/rf_yandex_results.csv`, leaving the baseline at
`benchmarks/rf_yandex_threshold_baseline.csv` unchanged.

The default sweep benchmarks these `(trees, depth, leaf-votes)` configurations:

```text
(60, 14, 1), (60, 15, 1), (60, 16, 1)
(20, 14, 1), (20, 15, 1), (20, 16, 1)
```

The probability thresholds range from `5e-7` to `2e-5`. Each point uses the first 1,000 test
queries, 100 warmup queries, and one timed repetition. Override these defaults with
`--votes-required`, `--queries`, `--warmup`, and `--query-repeats`. Use `--output` to give an
experiment its own CSV and `--label` to name it in the generated comparison plot.

The results are written to:

```text
benchmarks/rf_yandex_threshold_baseline.csv  # preserved baseline
benchmarks/rf_yandex_results.csv             # default output for new runs
```

The plotter always includes the baseline and shows only each result set's global Pareto frontier.
Add one or more compatible experiment CSVs by repeating `--series`:

```bash
python3 benchmarks/plot_rf_yandex_pareto.py \
  --series 'experiment=benchmarks/experiment.csv'
```

The runner requires a C++17 compiler, OpenMP, HDF5 development files discoverable through
`pkg-config`, and Matplotlib. Use `--skip-build` to reuse `benchmarks/.build/mlann_example`, or
`--no-plot` to produce only the result CSV.
