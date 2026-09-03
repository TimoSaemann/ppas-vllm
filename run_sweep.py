import argparse
import itertools
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Run benchmark.py over combinations of models, prompt "
            "lengths, output lengths, seeds, loads and vLLM configurations."
        )
    )

    # ------------------------------------------------------------------
    # Benchmark script and models
    # ------------------------------------------------------------------

    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=Path("benchmark.py"),
        help="Path to benchmark.py.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen_3b"],
        help="One or more model aliases understood by benchmark.py.",
    )

    # ------------------------------------------------------------------
    # Request shapes
    # ------------------------------------------------------------------

    parser.add_argument(
        "--prompt-lengths",
        type=comma_separated_ints,
        default=[25000],
        metavar="TOKENS",
        help=(
            "Comma-separated prompt lengths. "
            "Example: --prompt-lengths 20000,25000"
        ),
    )

    parser.add_argument(
        "--output-lengths",
        type=comma_separated_ints,
        default=[32],
        metavar="TOKENS",
        help=(
            "Comma-separated output lengths. "
            "Example: --output-lengths 16,32,64"
        ),
    )

    parser.add_argument(
        "--seeds",
        type=comma_separated_ints,
        default=[0],
        metavar="SEEDS",
        help=(
            "Comma-separated trace seeds. "
            "Example: --seeds 0,1,2,3,4"
        ),
    )

    # ------------------------------------------------------------------
    # Workload definition
    # ------------------------------------------------------------------

    parser.add_argument(
        "--arrival-mode",
        choices=["deterministic", "piecewise_poisson"],
        default="piecewise_poisson",
        help="Request arrival process.",
    )

    parser.add_argument(
        "--request-rates",
        type=comma_separated_floats,
        default=[1.0],
        metavar="RATES",
        help=(
            "Comma-separated request rates for deterministic workloads. "
            "Example: --request-rates 0.5,1.0,2.0,4.0"
        ),
    )

    parser.add_argument(
        "--duration",
        type=comma_separated_floats,
        default=[10.0],
        metavar="SECONDS",
        help=(
            "Comma-separated durations for deterministic workloads. "
            "Example: --duration 10,20,30,60"
        ),
    )

    parser.add_argument(
        "--phases",
        nargs="+",
        type=validate_phase_schedule,
        default=[
            "10:0.2,10:1.0,10:0.2,10:1.0,10:0.2",
        ],
        help=(
            "Piecewise phase schedules. Pass each complete schedule as a "
            "separate argument. Example: "
            "--phases "
            "'10:0.2,10:0.8,10:0.2,10:0.8,10:0.2' "
            "'10:0.2,10:1.2,10:0.2,10:1.2,10:0.2'"
        ),
    )
    parser.add_argument(
        "--mixed-workload",
        action="store_true",
        help="Use the heterogeneous 20-shape long-context workload defined in benchmark.py.",
    )

    # ------------------------------------------------------------------
    # vLLM configurations forwarded to benchmark.py
    # ------------------------------------------------------------------

    parser.add_argument(
        "--configs",
        type=str,
        default="2048,16384",
        help=(
            "Comma-separated benchmark configurations forwarded to "
            "benchmark.py. Example: "
            "--configs ppas_plus_nemotron,ppas_nemotron,"
            "1024,1280,2048,4096,8192,16384"
        ),
    )

    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="max_num_seqs forwarded to benchmark.py.",
    )

    # ------------------------------------------------------------------
    # Profiling and server behavior
    # ------------------------------------------------------------------

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Forward --profile to benchmark.py.",
    )

    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Optional Nsight Systems profile directory.",
    )

    parser.add_argument(
        "--reuse-server",
        action="store_true",
        help="Forward --reuse-server to benchmark.py.",
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Forward --skip-warmup to benchmark.py.",
    )

    # ------------------------------------------------------------------
    # Sweep execution
    # ------------------------------------------------------------------

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue with the next experiment if one benchmark invocation "
            "fails. By default, the sweep stops on the first failure."
        ),
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Optional file receiving the complete combined stdout/stderr "
            "of the sweep. Output is still shown in the terminal."
        ),
    )

    parser.add_argument(
        "--debug-trace",
        action="store_true",
        help="Enable detailed scheduler debug tracing.",
    )

    return parser.parse_args()


def comma_separated_ints(value: str) -> list[int]:
    try:
        values = [
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integers, got {value!r}."
        ) from exc

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one integer value is required."
        )

    return values


def comma_separated_floats(value: str) -> list[float]:
    try:
        values = [
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated numbers, got {value!r}."
        ) from exc

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one numeric value is required."
        )

    return values


def validate_phase_schedule(value: str) -> str:
    items = [item.strip() for item in value.split(",") if item.strip()]

    if not items:
        raise argparse.ArgumentTypeError(
            "A phase schedule must contain at least one phase."
        )

    for item in items:
        try:
            duration_str, rate_str = item.split(":", maxsplit=1)
            duration = float(duration_str)
            rate = float(rate_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid phase {item!r}; expected duration:rate."
            ) from exc

        if duration <= 0:
            raise argparse.ArgumentTypeError(
                f"Phase duration must be positive: {item!r}."
            )

        if rate < 0:
            raise argparse.ArgumentTypeError(
                f"Request rate must be non-negative: {item!r}."
            )

    return value


def sanitize(value: str) -> str:
    return (
        value.replace(":", "-")
        .replace(",", "_")
        .replace(".", "p")
        .replace("/", "-")
        .replace(" ", "")
    )


def build_cases(
    args: argparse.Namespace,
) -> Iterable[
    tuple[str, int, int, int, str | float, float | None]
]:
    if args.arrival_mode == "piecewise_poisson":
        return itertools.product(
            args.models,
            args.prompt_lengths,
            args.output_lengths,
            args.seeds,
            args.phases,
            [None],
        )

    if args.arrival_mode == "deterministic":
        return itertools.product(
            args.models,
            args.prompt_lengths,
            args.output_lengths,
            args.seeds,
            args.request_rates,
            args.duration,
        )

    raise ValueError(
        f"Unsupported arrival mode: {args.arrival_mode}"
    )


class OutputWriter:
    """Write text to stdout and optionally mirror it to a log file."""

    def __init__(self, log_file: Path | None) -> None:
        self._file = None

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._file = log_file.open(
                mode="w",
                encoding="utf-8",
                buffering=1,
            )

    def write(self, text: str) -> None:
        print(text, end="", flush=True)

        if self._file is not None:
            self._file.write(text)
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def format_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run_command(
    cmd: list[str],
    writer: OutputWriter,
) -> int:

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError("Could not capture benchmark output.")

    for line in process.stdout:
        writer.write(line)

    return process.wait()


def run_sweep(
    args: argparse.Namespace,
    writer: OutputWriter,
) -> tuple[int, int]:
    arrival_mode = args.arrival_mode
    succeeded = 0
    failed = 0

    configs = [
        config.strip()
        for config in args.configs.split(",")
        if config.strip()
    ]

    if not configs:
        raise ValueError("At least one benchmark configuration is required.")

    for (
        model,
        prompt_tokens,
        output_tokens,
        seed,
        workload_value,
        duration,
    ) in build_cases(args):

        for config in configs:
            cmd = [
                sys.executable,
                str(args.benchmark_script),
                "--model",
                model,
                "--prompt-tokens",
                str(prompt_tokens),
                "--output-tokens",
                str(output_tokens),
                "--arrival-mode",
                arrival_mode,
                "--seed",
                str(seed),
                "--configs",
                config,
                "--max-num-seqs",
                str(args.max_num_seqs),
            ]

            if arrival_mode == "piecewise_poisson":
                phases = str(workload_value)

                experiment_name = (
                    f"piecewise"
                    f"_model-{sanitize(model)}"
                    f"_p{prompt_tokens}"
                    f"_o{output_tokens}"
                    f"_phases-{sanitize(phases)}"
                    f"_seed{seed}"
                    f"_config-{sanitize(config)}"
                )

                cmd.extend([
                    "--phases",
                    phases,
                ])

            else:
                rate = float(workload_value)

                if duration is None:
                    raise ValueError(
                        "Deterministic workload is missing duration."
                    )

                experiment_name = (
                    f"{arrival_mode}"
                    f"_model-{sanitize(model)}"
                    f"_p{prompt_tokens}"
                    f"_o{output_tokens}"
                    f"_r{rate:g}"
                    f"_d{duration:g}"
                    f"_seed{seed}"
                    f"_config-{sanitize(config)}"
                )

                cmd.extend([
                    "--request-rate",
                    str(rate),
                    "--duration",
                    str(duration),
                ])

            if args.profile:
                cmd.append("--profile")

            if args.profile_dir is not None:
                cmd.extend([
                    "--profile-dir",
                    str(args.profile_dir),
                ])

            if args.reuse_server:
                cmd.append("--reuse-server")

            if args.skip_warmup:
                cmd.append("--skip-warmup")

            if args.debug_trace:
                cmd.append("--debug-trace")

            if args.mixed_workload:
                cmd.append("--mixed-workload")

            writer.write("=" * 100 + "\n")
            writer.write(f"EXPERIMENT: {experiment_name}\n")
            writer.write(f"RUN: {format_command(cmd)}\n")
            writer.write("=" * 100 + "\n")

            return_code = run_command(
                cmd=cmd,
                writer=writer,
            )

            if return_code == 0:
                succeeded += 1
                continue

            failed += 1

            writer.write(
                f"\nERROR: Experiment {experiment_name!r} exited with "
                f"status {return_code}.\n"
            )

            if not args.continue_on_error:
                raise subprocess.CalledProcessError(
                    returncode=return_code,
                    cmd=cmd,
                )

    return succeeded, failed


def main() -> None:
    args = parse_args()

    if args.max_num_seqs <= 0:
        raise SystemExit("--max-num-seqs must be positive.")

    if any(value <= 0 for value in args.prompt_lengths):
        raise SystemExit(
            "--prompt-lengths must contain positive values."
        )

    if any(value <= 0 for value in args.output_lengths):
        raise SystemExit(
            "--output-lengths must contain positive values."
        )

    if args.arrival_mode != "piecewise_poisson":
        if any(rate <= 0 for rate in args.request_rates):
            raise SystemExit(
                "--request-rates must contain positive values."
            )

        if any(duration <= 0 for duration in args.duration):
            raise SystemExit(
                "--duration must contain positive values."
            )

    if not args.benchmark_script.exists():
        raise SystemExit(
            f"Benchmark script not found: {args.benchmark_script}"
        )

    writer = OutputWriter(args.log_file)

    try:
        writer.write("SWEEP CONFIGURATION\n")
        writer.write(f"  models          : {args.models}\n")
        writer.write(f"  prompt lengths  : {args.prompt_lengths}\n")
        writer.write(f"  output lengths  : {args.output_lengths}\n")
        writer.write(f"  seeds           : {args.seeds}\n")
        writer.write(f"  configs         : {args.configs}\n")
        writer.write(f"  max num seqs    : {args.max_num_seqs}\n")
        writer.write(f"  arrival mode    : {args.arrival_mode}\n")

        if args.arrival_mode == "piecewise_poisson":
            writer.write(
                f"  phases          : {args.phases}\n"
            )
        else:
            writer.write(
                f"  request rates   : {args.request_rates}\n"
            )
            writer.write(
                f"  duration        : {args.duration}\n"
            )

        writer.write("\n")

        succeeded, failed = run_sweep(
            args=args,
            writer=writer,
        )

        writer.write("\n" + "=" * 100 + "\n")
        writer.write("SWEEP COMPLETE\n")
        writer.write(
            f"Successful experiments: {succeeded}\n"
        )
        writer.write(
            f"Failed experiments    : {failed}\n"
        )
        writer.write("=" * 100 + "\n")

    finally:
        writer.close()


if __name__ == "__main__":
    main()

