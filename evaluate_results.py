#!/usr/bin/env python3
"""
Compare P-PAS and fixed-MBT vLLM scheduler benchmark results.

Expected input
--------------
A run_sweep.py log containing completed benchmark runs for both fixed-MBT
configurations and P-PAS. Each completed run must contain one Python
dictionary line beginning with:

    {'trace_id': ...}

The script:
- parses the combined log,
- labels ppas runs as "ppas",
- labels fixed-MBT runs from config_name / max_num_batched_tokens,
- groups by burst rate and scheduler,
- computes mean and sample standard deviation across seeds,
- checks that comparable scheduler configurations use matching seeds,
- prints summary tables,
- optionally writes a numeric CSV.

Example
-------
python evaluate_scheduler_comparison.py \
    --input results.txt

python evaluate_scheduler_comparison.py \
    --input results.txt \
    --csv scheduler_summary.csv
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_METRICS = [
    "num_requests",
    "request_rate",
    "peak_arrivals_1s",
    "avg_latency",
    "p95_latency",
    "avg_ttft",
    "p95_ttft",
    "avg_tpot",
    "p95_tpot",
    "makespan",
    "failed",
]


def parse_result_file(path: Path) -> pd.DataFrame:
    """Extract final benchmark-result dictionaries from a run_sweep.py log."""
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line.startswith("{'trace_id':"):
                continue

            try:
                record = ast.literal_eval(line)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Could not parse result dictionary in {path} "
                    f"at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a dictionary in {path} at line {line_number}, "
                    f"but found {type(record).__name__}."
                )

            record["_source_file"] = str(path)
            records.append(record)

    if not records:
        raise ValueError(
            f"No completed benchmark-result dictionaries were found in {path}."
        )

    frame = pd.DataFrame.from_records(records)

    required = {"phase_schedule", "seed"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required result fields: "
            + ", ".join(sorted(missing))
        )

    return frame


def extract_burst_rate(schedule: str) -> float:
    """
    Derive the high/burst arrival rate from a schedule such as:
    10:0.2,10:0.8,10:0.2,10:0.8,10:0.2
    """
    try:
        rates = [
            float(part.split(":", maxsplit=1)[1])
            for part in schedule.split(",")
        ]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse phase schedule: {schedule!r}") from exc

    return max(rates)


def label_scheduler(row: pd.Series) -> str:
    """
    Label a scheduler configuration from the benchmark result row.
    """
    config_name = str(row.get("config_name", ""))

    if config_name.startswith("ppas_"):
        return "ppas"

    value = row.get("max_num_batched_tokens")

    if pd.notna(value):
        try:
            return f"MBT {int(value) // 1024}k"
        except (TypeError, ValueError):
            pass

    if config_name.startswith("mbt_"):
        try:
            token_text = config_name.split("_", maxsplit=2)[1]
            return f"MBT {int(token_text) // 1024}k"
        except (IndexError, ValueError):
            pass

    raise ValueError(
        "Could not determine scheduler configuration from "
        f"max_num_batched_tokens={row.get('max_num_batched_tokens')!r}, "
        f"config_name={row.get('config_name')!r}."
    )


def prepare_input(path: Path) -> pd.DataFrame:
    """Parse and label benchmark runs from one combined run_sweep.py log."""
    frame = parse_result_file(path).copy()

    frame["scheduler"] = frame.apply(
        label_scheduler,
        axis=1,
    )

    frame["burst_rate"] = frame["phase_schedule"].map(
        extract_burst_rate
    )

    return frame


def check_duplicate_runs(frame: pd.DataFrame) -> None:
    """
    Detect accidental duplicate result entries for the same workload,
    scheduler, and seed.
    """
    key = ["phase_schedule", "scheduler", "seed"]
    counts = frame.groupby(key, dropna=False).size()
    duplicates = counts[counts > 1]

    if not duplicates.empty:
        details = "\n".join(
            f"  schedule={schedule}, scheduler={scheduler}, "
            f"seed={seed}: {count} entries"
            for (schedule, scheduler, seed), count in duplicates.items()
        )
        raise ValueError(
            "Duplicate benchmark entries were found:\n" + details
        )


def warn_about_seed_mismatches(frame: pd.DataFrame) -> None:
    """
    Warn when schedulers for the same workload do not use identical seed sets.

    Paired seeds are important because they ensure all schedulers see the same
    stochastic arrival traces.
    """
    warnings: list[str] = []

    for schedule, group in frame.groupby("phase_schedule", sort=True):
        seed_sets = {
            scheduler: set(rows["seed"].tolist())
            for scheduler, rows in group.groupby("scheduler")
        }

        if len(seed_sets) < 2:
            warnings.append(
                f"schedule={schedule}: only found "
                f"{', '.join(sorted(seed_sets))}"
            )
            continue

        unique_sets = {frozenset(seeds) for seeds in seed_sets.values()}
        if len(unique_sets) > 1:
            formatted = ", ".join(
                f"{scheduler}={sorted(seeds)}"
                for scheduler, seeds in sorted(seed_sets.items())
            )
            warnings.append(f"schedule={schedule}: {formatted}")

    if warnings:
        print(
            "Warning: scheduler configurations do not all use matching seeds:\n"
            + "\n".join(f"  {warning}" for warning in warnings)
            + "\n",
            file=sys.stderr,
        )


def aggregate(
    frame: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Compute mean and sample standard deviation across seeds."""
    available_metrics = [metric for metric in metrics if metric in frame.columns]

    if not available_metrics:
        raise ValueError("None of the requested metrics exists in the input files.")

    group_columns = ["burst_rate", "phase_schedule", "scheduler"]
    grouped = frame.groupby(group_columns, sort=True, dropna=False)

    summary = grouped[available_metrics].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()
    ]

    summary.insert(0, "n_seeds", grouped["seed"].nunique())
    summary = summary.reset_index()

    scheduler_order = {
        "MBT 1k": 0,
        "MBT 2k": 1,
        "MBT 4k": 2,
        "MBT 8k": 3,
        "MBT 16k": 4,
        "ppas": 5,
    }
    summary["_scheduler_order"] = summary["scheduler"].map(
        scheduler_order
    ).fillna(999)

    summary = summary.sort_values(
        ["burst_rate", "_scheduler_order", "scheduler"]
    ).drop(columns="_scheduler_order")

    return summary


def format_mean_std(
    summary: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Create a compact human-readable table with mean ± std cells."""
    output = summary[["burst_rate", "scheduler", "n_seeds"]].copy()

    precision = {
        "num_requests": 2,
        "request_rate": 3,
        "peak_arrivals_1s": 2,
        "avg_latency": 3,
        "p95_latency": 3,
        "avg_ttft": 3,
        "p95_ttft": 3,
        "avg_tpot": 4,
        "p95_tpot": 4,
        "makespan": 3,
        "failed": 2,
    }

    for metric in metrics:
        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"

        if mean_column not in summary.columns:
            continue

        digits = precision.get(metric, 3)

        def format_row(row: pd.Series) -> str:
            mean = row[mean_column]
            std = row[std_column]

            if pd.isna(std):
                return f"{mean:.{digits}f} ± n/a"

            return f"{mean:.{digits}f} ± {std:.{digits}f}"

        output[metric] = summary.apply(format_row, axis=1)

    return output

def print_full_table(summary: pd.DataFrame, metrics: list[str]) -> None:
    """Print all requested metrics."""
    display = format_mean_std(summary, metrics)
    print("\nALL METRICS")
    print(display.to_string(index=False))





def print_normalized_summary(summary, metrics=None):
    """
    Geometric mean of ppas vs static schedulers across burst rates.
    Each burst-rate contributes equally regardless of absolute latency.
    """

    if metrics is None:
        metrics = [
            "avg_latency",
            "p95_latency",
            "avg_ttft",
            "avg_tpot",
            "makespan",
        ]

    burst_rates = sorted(summary["burst_rate"].unique())

    print()
    print("=" * 80)
    print("GEOMETRIC MEAN OF NORMALIZED RATIOS")
    print("(Each burst-rate contributes equally.)")
    print("=" * 80)

    for baseline in ["MBT 2k", "MBT 16k"]:

        if baseline not in summary["scheduler"].values:
            continue

        print(f"\nP-PAS vs {baseline}")

        for metric in metrics:
            ratios = []

            for br in burst_rates:
                mean_col = f"{metric}_mean"

                base_rows = summary.loc[
                    (summary["scheduler"] == baseline)
                    & (summary["burst_rate"] == br),
                    mean_col,
                ]

                ppas_rows = summary.loc[
                    (summary["scheduler"] == "ppas")
                    & (summary["burst_rate"] == br),
                    mean_col,
                ]

                if base_rows.empty or ppas_rows.empty:
                    continue

                ratios.append(ppas_rows.iloc[0] / base_rows.iloc[0])

            if not ratios:
                continue

            geo_ratio = np.exp(np.mean(np.log(ratios)))
            improvement = (1.0 - geo_ratio) * 100

            print(
                f"  {metric:12s}: "
                f"ratio={geo_ratio:.3f}   "
                f"improvement={improvement:+.2f}%"
            )

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ppas and fixed-MBT vLLM scheduler results "
            "using mean and sample standard deviation across seeds."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Combined benchmark log containing fixed-MBT and "
            "P-PAS runs."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional output path for the numeric summary CSV.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to aggregate.",
    )
    parser.add_argument(
        "--burst-rates",
        type=float,
        nargs="+",
        default=None,
        help="Only evaluate these burst rates, e.g. --burst-rates 0.8 1.2.",
    )
    args = parser.parse_args()

    try:
        runs = prepare_input(args.input)
        if args.burst_rates is not None:
            runs = runs[runs["burst_rate"].isin(args.burst_rates)].copy()

            if runs.empty:
                raise ValueError(
                    f"No runs found for requested burst rates: {args.burst_rates}"
                )
        check_duplicate_runs(runs)
        warn_about_seed_mismatches(runs)
        summary = aggregate(runs, args.metrics)

    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Parsed {len(runs)} runs from {args.input}.\n"
        f"Workloads: {runs['phase_schedule'].nunique()}\n"
        f"Schedulers: {', '.join(sorted(runs['scheduler'].unique()))}\n"
        f"Seeds: {', '.join(map(str, sorted(runs['seed'].unique())))}\n"
    )

    print_full_table(summary, args.metrics)
    print_normalized_summary(summary)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.csv, index=False)
        print(f"\nSaved numeric summary to: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
