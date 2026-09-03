#!/usr/bin/env python3
"""
Evaluate fixed-MBT, P-PAS, and P-PAS+ vLLM benchmark results.

Expected input
--------------
A run_sweep.py log containing aggregate benchmark results and,
optionally, REQUEST_RESULT entries for exact per-request analysis.

The script:
- parses aggregate benchmark runs,
- labels fixed-MBT, P-PAS, and P-PAS+ configurations,
- aggregates metrics across seeds,
- checks scheduler seed consistency,
- prints the main P-PAS+ workload-level summary,
- optionally evaluates per-request and request-type results,
- optionally writes numeric CSV summaries.

Example
-------
python evaluate_results.py \
    --input results/nemotron/ppas_hetero_final.txt \
    --request-types
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
    #"failed",
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


def parse_request_results(path: Path) -> pd.DataFrame:
    """Extract per-request dictionaries from REQUEST_RESULT lines."""
    records: list[dict[str, Any]] = []
    prefix = "REQUEST_RESULT:"

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line.startswith(prefix):
                continue

            text = line[len(prefix):].strip()

            try:
                record = ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Could not parse REQUEST_RESULT in {path} "
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
            f"No REQUEST_RESULT entries were found in {path}."
        )

    frame = pd.DataFrame.from_records(records)

    required = {
        "phase_schedule",
        "seed",
        "requested_prompt_tokens",
        "requested_output_tokens",
        "latency",
        "ttft",
        "tpot",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required REQUEST_RESULT fields: "
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
        raise ValueError(
            f"Could not parse phase schedule: {schedule!r}"
        ) from exc

    return max(rates)


def extract_burst_duration(schedule: str) -> float:
    """
    Derive the duration of the highest-rate burst phase.

    Example:
        20:0.1,40:0.8,20:0.1 -> 40.0
    """
    try:
        phases = []

        for part in schedule.split(","):
            duration_text, rate_text = part.split(":", maxsplit=1)

            phases.append(
                (
                    float(duration_text),
                    float(rate_text),
                )
            )

    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Could not parse phase schedule: {schedule!r}"
        ) from exc

    burst_rate = max(
        rate
        for _, rate in phases
    )

    burst_durations = [
        duration
        for duration, rate in phases
        if rate == burst_rate
    ]

    return sum(burst_durations)


def label_scheduler(row: pd.Series) -> str:
    config_name = str(row.get("config_name", ""))

    if config_name.startswith("ppas_plus_"):
        return "ppas_plus"

    if config_name.startswith("ppas_"):
        return "ppas"

    value = row.get("max_num_batched_tokens")
    if pd.notna(value):
        try:
            return f"MBT {int(value)}"
        except (TypeError, ValueError):
            pass

    if config_name.startswith("mbt_"):
        try:
            token_text = config_name.split("_", maxsplit=2)[1]
            return f"MBT {int(token_text)}"
        except (IndexError, ValueError):
            pass

    raise ValueError(
        f"Could not determine scheduler label from config_name={config_name!r}"
    )


def scheduler_sort_key(name: str) -> tuple[int, float, float]:
    if name.startswith("MBT "):
        try:
            return 0, float(name.split()[1]), 0.0
        except (IndexError, ValueError):
            return 99, 0.0, 0.0

    if name == "ppas":
        return 1, 0.0, 0.0

    if name == "ppas_plus":
        return 2, 0.0, 0.0

    return 99, 0.0, 0.0


def get_load_rate(row: pd.Series) -> float:
    if row["arrival_mode"] == "piecewise_poisson":
        return extract_burst_rate(row["phase_schedule"])

    if row["arrival_mode"] == "deterministic":
        return float(row["configured_request_rate"])

    raise ValueError(
        f"Unsupported arrival mode: {row['arrival_mode']!r}"
    )

def get_duration(row):
    if row["arrival_mode"] == "piecewise_poisson":
        return sum(
            float(phase.split(":")[0])
            for phase in row["phase_schedule"].split(",")
        )

    return (
        row["num_requests"]
        / float(row["configured_request_rate"])
    )

def prepare_input(path: Path) -> pd.DataFrame:
    """Parse and label benchmark runs from one combined run_sweep.py log."""
    frame = parse_result_file(path).copy()

    frame["scheduler"] = frame.apply(
        label_scheduler,
        axis=1,
    )

    frame["duration"] = frame.apply(get_duration, axis=1)

    frame["burst_rate"] = frame.apply(
        get_load_rate,
        axis=1,
    )

    frame["burst_duration"] = frame["phase_schedule"].map(
        extract_burst_duration
    )

    return frame


def prepare_request_input(path: Path) -> pd.DataFrame:
    """Parse and label per-request benchmark results."""
    frame = parse_request_results(path).copy()

    frame["scheduler"] = frame.apply(
        label_scheduler,
        axis=1,
    )

    frame["burst_rate"] = frame["phase_schedule"].map(
        extract_burst_rate
    )

    frame["burst_duration"] = frame["phase_schedule"].map(
        extract_burst_duration
    )

    frame["request_type"] = (
            frame["requested_prompt_tokens"].astype(int).astype(str)
            + "/"
            + frame["requested_output_tokens"].astype(int).astype(str)
    )

    return frame


def collapse_duplicate_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse repeated executions of the same scheduler/workload/seed.

    Numeric result fields are averaged across repeats. Non-numeric metadata is
    taken from the first row. The returned frame therefore contains exactly one
    row per workload/scheduler/seed.
    """
    key = [
        "arrival_mode",
        "configured_prompt_tokens",
        "configured_output_tokens",
        "duration",
        "burst_rate",
        "scheduler",
        "seed",
    ]

    counts = frame.groupby(
        key,
        dropna=False,
    ).size()

    duplicates = counts[counts > 1]

    if duplicates.empty:
        return frame

    print(
        "Note: repeated benchmark executions were found and will be averaged "
        "within each scheduler/workload/seed before seed aggregation:",
        file=sys.stderr,
    )

    for (
            arrival_mode,
            prompt_tokens,
            output_tokens,
            duration,
            burst_rate,
            scheduler,
            seed,
    ), count in duplicates.items():
        print(
            f"  arrival_mode={arrival_mode}, "
            f"prompt={prompt_tokens}, "
            f"output={output_tokens}, "
            f"duration={duration}, "
            f"burst_rate={burst_rate}, "
            f"scheduler={scheduler}, "
            f"seed={seed}: {count} repeats",
            file=sys.stderr,
        )

    numeric_columns = set(
        frame.select_dtypes(include=[np.number]).columns
    )
    numeric_columns.difference_update(key)

    aggregations: dict[str, str] = {}

    for column in frame.columns:
        if column in key:
            continue

        if column in numeric_columns:
            aggregations[column] = "mean"
        else:
            aggregations[column] = "first"

    return (
        frame.groupby(
            key,
            sort=False,
            dropna=False,
            as_index=False,
        )
        .agg(aggregations)
    )

def warn_about_seed_mismatches(frame: pd.DataFrame) -> None:
    """
    Warn when schedulers for the same workload do not use identical seed sets.

    Paired seeds are important because they ensure all schedulers see the same
    stochastic arrival traces.
    """
    warnings: list[str] = []

    for schedule, group in frame.groupby(
        "phase_schedule",
        sort=True,
    ):
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

        unique_sets = {
            frozenset(seeds)
            for seeds in seed_sets.values()
        }

        if len(unique_sets) > 1:
            formatted = ", ".join(
                f"{scheduler}={sorted(seeds)}"
                for scheduler, seeds in sorted(seed_sets.items())
            )

            warnings.append(
                f"schedule={schedule}: {formatted}"
            )

    if warnings:
        print(
            "Warning: scheduler configurations do not all use matching seeds:\n"
            + "\n".join(
                f"  {warning}"
                for warning in warnings
            )
            + "\n",
            file=sys.stderr,
        )


def warn_about_request_type_seed_mismatches(
    frame: pd.DataFrame,
) -> None:
    """
    Warn if some request types are absent for some scheduler/seed combinations.
    """
    warnings: list[str] = []

    for (schedule, request_type), group in frame.groupby(
        ["phase_schedule", "request_type"],
        sort=True,
    ):
        seed_sets = {
            scheduler: set(rows["seed"].tolist())
            for scheduler, rows in group.groupby("scheduler")
        }

        if len(seed_sets) < 2:
            continue

        unique_sets = {
            frozenset(seeds)
            for seeds in seed_sets.values()
        }

        if len(unique_sets) > 1:
            formatted = ", ".join(
                f"{scheduler}={sorted(seeds)}"
                for scheduler, seeds in sorted(seed_sets.items())
            )

            warnings.append(
                f"schedule={schedule}, "
                f"request_type={request_type}: {formatted}"
            )

    if warnings:
        print(
            "Warning: request-type scheduler groups do not all use matching "
            "seed sets:\n"
            + "\n".join(
                f"  {warning}"
                for warning in warnings
            )
            + "\n",
            file=sys.stderr,
        )


def aggregate(
    frame: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Compute mean and sample standard deviation across seeds."""
    available_metrics = [
        metric
        for metric in metrics
        if metric in frame.columns
    ]

    if not available_metrics:
        raise ValueError(
            "None of the requested metrics exists in the input files."
        )

    group_columns = [
        "configured_prompt_tokens",
        "configured_output_tokens",
        "duration",
        "burst_rate",
        "burst_duration",
        "scheduler",
    ]

    grouped = frame.groupby(
        group_columns,
        sort=True,
        dropna=False,
    )

    summary = grouped[available_metrics].agg(
        ["mean", "std"]
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns.to_flat_index()
    ]

    summary.insert(
        0,
        "n_seeds",
        grouped["seed"].nunique(),
    )

    summary = summary.reset_index()

    summary["_scheduler_order"] = summary["scheduler"].map(
        lambda name: scheduler_sort_key(name)[0]
    )

    summary["_scheduler_value"] = summary["scheduler"].map(
        lambda name: scheduler_sort_key(name)[1]
    )

    summary = summary.sort_values(
        [
            "configured_prompt_tokens",
            "configured_output_tokens",
            "burst_rate",
            "burst_duration",
            "_scheduler_order",
            "_scheduler_value",
            "scheduler",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
    ).drop(
        columns=[
            "_scheduler_order",
            "_scheduler_value",
        ]
    )

    return summary


def aggregate_request_types(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate mixed-workload metrics in two stages:

    1. Compute request-type metrics within each seed.
    2. Compute mean and sample std across seeds.

    This prevents seeds that happen to contain more requests of one type from
    receiving more weight in the final summary.
    """
    per_seed = (
        frame.groupby(
            [
                "burst_rate",
                "burst_duration",
                "phase_schedule",
                "scheduler",
                "seed",
                "request_type",
                "requested_prompt_tokens",
                "requested_output_tokens",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            num_requests=("latency", "count"),
            avg_latency=("latency", "mean"),
            p95_latency=(
                "latency",
                lambda x: x.quantile(0.95),
            ),
            avg_ttft=("ttft", "mean"),
            p95_ttft=(
                "ttft",
                lambda x: x.quantile(0.95),
            ),
            avg_tpot=("tpot", "mean"),
            p95_tpot=(
                "tpot",
                lambda x: x.quantile(0.95),
            ),
        )
        .reset_index()
    )

    metrics = [
        "num_requests",
        "avg_latency",
        "p95_latency",
        "avg_ttft",
        "p95_ttft",
        "avg_tpot",
        "p95_tpot",
    ]

    group_columns = [
        "burst_rate",
        "burst_duration",
        "phase_schedule",
        "scheduler",
        "request_type",
        "requested_prompt_tokens",
        "requested_output_tokens",
    ]

    grouped = per_seed.groupby(
        group_columns,
        sort=True,
        dropna=False,
    )

    summary = grouped[metrics].agg(
        ["mean", "std"]
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns.to_flat_index()
    ]

    summary.insert(
        0,
        "n_seeds",
        grouped["seed"].nunique(),
    )

    summary = summary.reset_index()

    summary["_scheduler_order"] = summary["scheduler"].map(
        lambda name: scheduler_sort_key(name)[0]
    )

    summary["_scheduler_value"] = summary["scheduler"].map(
        lambda name: scheduler_sort_key(name)[1]
    )

    summary = summary.sort_values(
        [
            "burst_rate",
            "burst_duration",
            "requested_prompt_tokens",
            "requested_output_tokens",
            "_scheduler_order",
            "_scheduler_value",
            "scheduler",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            True,
            True,
            True,
        ],
    ).drop(
        columns=[
            "_scheduler_order",
            "_scheduler_value",
        ]
    )

    return per_seed, summary

def format_mean_std(
    summary: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Create a compact human-readable table with mean ± std cells."""
    output = summary[
        [
            "configured_prompt_tokens",
            "configured_output_tokens",
            "duration",
            "burst_rate",
            "burst_duration",
            "scheduler",
            "n_seeds",
        ]
    ].copy()

    precision = {
        "num_requests": 2,
        "request_rate": 3,
        "peak_arr_1s": 2,
        "avg_latency": 3,
        "p95_latency": 3,
        "avg_ttft": 3,
        "p95_ttft": 3,
        "avg_tpot": 4,
        "p95_tpot": 4,
        "makespan": 3,
        #"failed": 2,
    }

    for metric in metrics:
        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"

        if mean_column not in summary.columns:
            continue

        digits = precision.get(
            metric,
            3,
        )

        def format_row(row: pd.Series) -> str:
            mean_value = row[mean_column]
            std_value = row[std_column]

            if pd.isna(std_value):
                return (
                    f"{mean_value:.{digits}f} ± n/a"
                )

            return (
                f"{mean_value:.{digits}f} ± "
                f"{std_value:.{digits}f}"
            )

        output[metric] = summary.apply(
            format_row,
            axis=1,
        )

    return output


def format_request_type_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact mean ± std table for request-type metrics."""
    output = summary[
        [
            "burst_rate",
            "burst_duration",
            "request_type",
            "scheduler",
            "n_seeds",
        ]
    ].copy()

    precision = {
        "num_requests": 2,
        "avg_latency": 3,
        "p95_latency": 3,
        "avg_ttft": 3,
        "p95_ttft": 3,
        "avg_tpot": 4,
        "p95_tpot": 4,
    }

    for metric, digits in precision.items():
        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"

        if mean_column not in summary.columns:
            continue

        def format_row(row: pd.Series) -> str:
            mean_value = row[mean_column]
            std_value = row[std_column]

            if pd.isna(std_value):
                return (
                    f"{mean_value:.{digits}f} ± n/a"
                )

            return (
                f"{mean_value:.{digits}f} ± "
                f"{std_value:.{digits}f}"
            )

        output[metric] = summary.apply(
            format_row,
            axis=1,
        )

    return output


def print_full_table(
    summary: pd.DataFrame,
    metrics: list[str],
) -> None:
    """Print all requested aggregate metrics."""
    display = format_mean_std(
        summary,
        metrics,
    )

    print("\nALL METRICS")
    print(
        display.to_string(
            index=False,
        )
    )

def print_request_type_summary(
    summary: pd.DataFrame,
) -> None:
    """Print mixed-workload metrics separately per request type."""
    display = format_request_type_summary(
        summary
    )

    print()
    print("=" * 110)
    print("METRICS BY REQUEST TYPE")
    print("=" * 110)
    print(
        display.to_string(
            index=False,
        )
    )

def build_paired_request_comparison(
    frame: pd.DataFrame,
    baseline: str,
    candidate: str = "ppas",
) -> pd.DataFrame:
    """
    Pair exact requests across two schedulers.

    Pairing uses workload, seed, request type, and request_id when available.
    This is stricter and fairer than comparing independently aggregated request
    distributions because both schedulers are evaluated on the same stochastic
    trace and the same logical request.

    Returns one row per matched request with latency/TTFT/TPOT ratios.
    A ratio below 1 means the candidate is better.
    """
    required = {
        "phase_schedule",
        "seed",
        "scheduler",
        "latency",
        "ttft",
        "tpot",
        "requested_prompt_tokens",
        "requested_output_tokens",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Cannot build paired request comparison; missing fields: "
            + ", ".join(sorted(missing))
        )

    pair_keys = [
        "phase_schedule",
        "burst_rate",
        "burst_duration",
        "seed",
        "requested_prompt_tokens",
        "requested_output_tokens",
    ]

    # request_id is the preferred exact pairing key. If it is absent, use
    # arrival time when available. We deliberately do not silently pair only by
    # request type because a trace may contain several requests of the same type.
    if "request_id" in frame.columns:
        pair_keys.append("request_id")
    elif "arrival_s" in frame.columns:
        pair_keys.append("arrival_s")
    elif "arrival_time" in frame.columns:
        pair_keys.append("arrival_time")
    else:
        raise ValueError(
            "Exact per-request pairing requires REQUEST_RESULT to contain "
            "'request_id', 'arrival_s', or 'arrival_time'."
        )

    subset = frame.loc[
        frame["scheduler"].isin([baseline, candidate])
    ].copy()

    counts = (
        subset.groupby(pair_keys + ["scheduler"], dropna=False)
        .size()
    )
    duplicate_pairs = counts[counts > 1]
    if not duplicate_pairs.empty:
        raise ValueError(
            "Exact request-pair keys are not unique for "
            f"{baseline!r} vs {candidate!r}. "
            "Check request_id/arrival fields or duplicate benchmark runs."
        )

    metrics = ["latency", "ttft", "tpot"]

    wide = subset.pivot(
        index=pair_keys,
        columns="scheduler",
        values=metrics,
    )

    if baseline not in wide.columns.get_level_values(1):
        return pd.DataFrame()

    if candidate not in wide.columns.get_level_values(1):
        return pd.DataFrame()

    paired = pd.DataFrame(index=wide.index).reset_index()

    for metric in metrics:
        paired[f"{metric}_baseline"] = wide[(metric, baseline)].to_numpy()
        paired[f"{metric}_candidate"] = wide[(metric, candidate)].to_numpy()

    paired = paired.dropna(
        subset=[
            "latency_baseline",
            "latency_candidate",
        ]
    ).copy()

    paired["request_type"] = (
        paired["requested_prompt_tokens"].astype(int).astype(str)
        + "/"
        + paired["requested_output_tokens"].astype(int).astype(str)
    )

    for metric in metrics:
        baseline_col = f"{metric}_baseline"
        candidate_col = f"{metric}_candidate"
        ratio_col = f"{metric}_ratio"

        valid = (
            paired[baseline_col].notna()
            & paired[candidate_col].notna()
            & (paired[baseline_col] > 0)
            & (paired[candidate_col] > 0)
        )

        paired[ratio_col] = np.nan
        paired.loc[valid, ratio_col] = (
            paired.loc[valid, candidate_col]
            / paired.loc[valid, baseline_col]
        )

    return paired

def find_best_static_scheduler(
    summary: pd.DataFrame,
) -> str | None:
    static_schedulers = [
        scheduler
        for scheduler in summary["scheduler"].unique()
        if scheduler.startswith("MBT ")
    ]

    if not static_schedulers:
        return None

    means = {}

    for scheduler in static_schedulers:
        values = summary.loc[
            summary["scheduler"] == scheduler,
            "avg_latency_mean",
        ]

        if values.empty:
            continue

        means[scheduler] = float(values.mean())

    if not means:
        return None

    return min(
        means,
        key=means.get,
    )

def print_ppas_plus_request_summary(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """
    Print the main exact paired per-request P-PAS+ results.

    Only E2E latency is reported here. P95 latency and makespan are
    workload-level metrics and are reported in the workload summary.
    """
    available_schedulers = set(frame["scheduler"])

    if "ppas_plus" not in available_schedulers:
        return

    best_static = find_best_static_scheduler(
        summary
    )

    if best_static is None:
        return

    def print_comparison(
        baseline: str,
        candidate: str,
        label: str,
    ) -> None:
        paired = build_paired_request_comparison(
            frame,
            baseline=baseline,
            candidate=candidate,
        )

        if paired.empty:
            return

        baseline_col = "latency_baseline"
        candidate_col = "latency_candidate"

        valid = paired.loc[
            paired[baseline_col].notna()
            & paired[candidate_col].notna()
            & (paired[baseline_col] > 0)
            & (paired[candidate_col] > 0)
        ].copy()

        if valid.empty:
            return

        # Ratio of the actual aggregate mean request latencies.
        ratio = float(
            valid[candidate_col].mean()
            / valid[baseline_col].mean()
        )

        improvement = (
            1.0 - ratio
        ) * 100.0

        wins = int(
            (
                valid[candidate_col]
                < valid[baseline_col]
            ).sum()
        )

        win_rate = (
            wins / len(valid)
        ) * 100.0

        print()
        print(label)
        print(
            f"  Matched requests     : {len(valid)}"
        )
        print(
            f"  Mean E2E improvement : {improvement:+.2f}%"
        )
        print(
            f"  Requests faster      : {win_rate:.1f}%"
        )

    print()
    print("=" * 80)
    print("PAIRED REQUEST SUMMARY")
    print("=" * 80)

    print_comparison(
        baseline=best_static,
        candidate="ppas_plus",
        label=f"P-PAS+ vs best static {best_static}",
    )

    if "ppas" in available_schedulers:
        print_comparison(
            baseline="ppas",
            candidate="ppas_plus",
            label="P-PAS+ vs P-PAS",
        )

def print_ppas_plus_workload_summary(
    summary: pd.DataFrame,
) -> None:
    """
    Print the main workload-level P-PAS+ results.

    Each workload is first aggregated across seeds. Static MBTs are compared
    using the arithmetic mean of per-workload candidate/baseline ratios.
    """
    workload_keys = [
        "configured_prompt_tokens",
        "configured_output_tokens",
        "duration",
        "burst_rate",
        "burst_duration",
    ]

    available_schedulers = set(summary["scheduler"])

    if "ppas_plus" not in available_schedulers:
        return

    static_schedulers = sorted(
        [
            scheduler
            for scheduler in available_schedulers
            if scheduler.startswith("MBT ")
        ],
        key=lambda name: scheduler_sort_key(name)[1],
    )

    def build_workload_pairs(
        baseline: str,
        candidate: str,
        metric: str,
    ) -> pd.DataFrame:
        mean_col = f"{metric}_mean"

        base = summary.loc[
            summary["scheduler"] == baseline,
            workload_keys + [mean_col],
        ].rename(
            columns={mean_col: "baseline_value"}
        )

        candidate_rows = summary.loc[
            summary["scheduler"] == candidate,
            workload_keys + [mean_col],
        ].rename(
            columns={mean_col: "candidate_value"}
        )

        paired = base.merge(
            candidate_rows,
            on=workload_keys,
            how="inner",
            validate="one_to_one",
        )

        return paired.loc[
            paired["baseline_value"].notna()
            & paired["candidate_value"].notna()
            & (paired["baseline_value"] > 0)
            & (paired["candidate_value"] > 0)
        ].copy()

    def summarize(
        baseline: str,
        candidate: str,
        metric: str,
    ) -> tuple[float, int, int]:
        paired = build_workload_pairs(
            baseline=baseline,
            candidate=candidate,
            metric=metric,
        )

        if paired.empty:
            return float("nan"), 0, 0

        ratios = (
            paired["candidate_value"]
            / paired["baseline_value"]
        )

        improvement = (
            1.0 - float(ratios.mean())
        ) * 100.0

        wins = int(
            (
                paired["candidate_value"]
                < paired["baseline_value"]
            ).sum()
        )

        return improvement, wins, len(paired)

    print()
    print("=" * 80)
    print("P-PAS+ SUMMARY (workload level)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # P-PAS+ against all static MBTs using mean E2E latency.
    # ------------------------------------------------------------------

    static_results = []

    for baseline in static_schedulers:
        improvement, wins, total = summarize(
            baseline=baseline,
            candidate="ppas_plus",
            metric="avg_latency",
        )

        if total == 0:
            continue

        static_results.append(
            (
                baseline,
                improvement,
                wins,
                total,
            )
        )

    if static_results:
        print()
        print("Mean E2E vs static MBT")

        for baseline, improvement, _, _ in static_results:
            print(
                f"  {baseline:10s} "
                f"{improvement:+7.2f}%"
            )

        # Best static baseline = smallest improvement of P-PAS+ over it.
        best_static = find_best_static_scheduler(summary)

        if best_static is not None:
            print()
            print(f"P-PAS+ vs best static {best_static}")

            display_metrics = [
                ("avg_latency", "Mean E2E"),
                ("p95_latency", "P95 E2E"),
                ("makespan", "Makespan"),
            ]

            for metric, label in display_metrics:
                improvement, wins, total = summarize(
                    baseline=best_static,
                    candidate="ppas_plus",
                    metric=metric,
                )

                if total == 0:
                    continue

                print(
                    f"  {label:9s}: "
                    f"{improvement:+6.2f}%   "
                    f"workloads won: {wins}/{total}"
                )

    # ------------------------------------------------------------------
    # P-PAS+ against original P-PAS.
    # ------------------------------------------------------------------

    if "ppas" in available_schedulers:
        print()
        print("P-PAS+ vs P-PAS")

        display_metrics = [
            ("avg_latency", "Mean E2E"),
            ("p95_latency", "P95 E2E"),
            ("makespan", "Makespan"),
        ]

        for metric, label in display_metrics:
            improvement, wins, total = summarize(
                baseline="ppas",
                candidate="ppas_plus",
                metric=metric,
            )

            if total == 0:
                continue

            print(
                f"  {label:9s}: "
                f"{improvement:+6.2f}%   "
                f"workloads won: {wins}/{total}"
            )

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed-MBT, P-PAS, and P-PAS+ vLLM scheduler results."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Combined benchmark log containing fixed-MBT, P-PAS, "
            "and P-PAS+ runs."
        ),
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "Optional output path for the aggregate numeric summary CSV."
        ),
    )

    parser.add_argument(
        "--request-types-csv",
        type=Path,
        default=None,
        help=(
            "Optional output path for the per-request-type numeric "
            "summary CSV."
        ),
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
        help=(
            "Only evaluate these burst rates, "
            "e.g. --burst-rates 0.8 1.2."
        ),
    )

    parser.add_argument(
        "--request-types",
        action="store_true",
        help=(
            "Additionally evaluate REQUEST_RESULT entries separately "
            "by requested prompt/output token lengths."
        ),
    )

    args = parser.parse_args()

    try:
        runs = prepare_input(
            args.input
        )

        if args.burst_rates is not None:
            runs = runs[
                runs["burst_rate"].isin(
                    args.burst_rates
                )
            ].copy()

            if runs.empty:
                raise ValueError(
                    "No aggregate runs found for requested "
                    f"burst rates: {args.burst_rates}"
                )

        runs = collapse_duplicate_runs(
            runs
        )

        warn_about_seed_mismatches(
            runs
        )

        summary = aggregate(
            runs,
            args.metrics,
        )

    except (OSError, ValueError) as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Parsed {len(runs)} aggregate runs from {args.input}.\n"
        f"Workloads: {runs['phase_schedule'].nunique()}\n"
        f"Schedulers: "
        f"{', '.join(sorted(runs['scheduler'].unique()))}\n"
        f"Seeds: "
        f"{', '.join(map(str, sorted(runs['seed'].unique())))}\n"
    )

    print_full_table(
        summary,
        args.metrics,
    )

    print_ppas_plus_workload_summary(
        summary
    )

    if args.csv is not None:
        args.csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary.to_csv(
            args.csv,
            index=False,
        )

        print(
            f"\nSaved aggregate numeric summary to: {args.csv}"
        )

    if args.request_types:
        try:
            request_runs = prepare_request_input(
                args.input
            )

            if args.burst_rates is not None:
                request_runs = request_runs[
                    request_runs["burst_rate"].isin(
                        args.burst_rates
                    )
                ].copy()

                if request_runs.empty:
                    raise ValueError(
                        "No REQUEST_RESULT entries found for requested "
                        f"burst rates: {args.burst_rates}"
                    )

            warn_about_request_type_seed_mismatches(
                request_runs
            )

            _, request_summary = aggregate_request_types(
                request_runs
            )

        except (OSError, ValueError) as exc:
            print(
                f"Error while evaluating request types: {exc}",
                file=sys.stderr,
            )
            return 1

        num_schedulers = request_runs["scheduler"].nunique()
        num_logical_requests = (
            len(request_runs) // num_schedulers
            if num_schedulers > 0
            else 0
        )

        print(
            "\n"
            f"Parsed {len(request_runs)} request executions across "
            f"{num_schedulers} schedulers "
            f"({num_logical_requests} logical requests).\n"
            f"Request types: "
            f"{', '.join(sorted(request_runs['request_type'].unique()))}\n"
        )

        print_ppas_plus_request_summary(
            request_runs,
            summary,
        )

        print()
        print("=" * 80)
        print("REQUEST-TYPE DIAGNOSTICS")
        print("=" * 80)

        print_request_type_summary(
            request_summary
        )

        if args.request_types_csv is not None:
            args.request_types_csv.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            request_summary.to_csv(
                args.request_types_csv,
                index=False,
            )

            print(
                "\nSaved per-request-type numeric summary to: "
                f"{args.request_types_csv}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
