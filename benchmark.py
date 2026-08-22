import argparse
import asyncio
import os
import random
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from openai import AsyncOpenAI
from transformers import AutoTokenizer, PreTrainedTokenizerBase

BASE_URL = "http://localhost:8000/v1"

MODELS = {
    "qwen_0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen_3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen_7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_27b": "unsloth/Qwen3.8-27B-NVFP4",
    "llama_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma_1b": "google/gemma-3-1b-it",
    "smollm3_3b": "HuggingFaceTB/SmolLM3-3B",
    "nemotron_30b": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Benchmark fixed-MBT and P-PAS scheduling policies "
            "for long-context vLLM serving."
        )
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run each vLLM server under Nsight Systems.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("nsys_profiles"),
    )

    parser.add_argument(
        "--reuse-server",
        action="store_true",
        help=(
            "Use an existing vLLM server at BASE_URL instead of starting "
            "and stopping a server."
        ),
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Do not send benchmark warm-up requests.",
    )

    parser.add_argument("--model", choices=MODELS.keys(), default="qwen_3b")

    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=24576,
    )

    parser.add_argument(
        "--output-tokens",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--request-rate",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--arrival-mode",
        choices=["deterministic", "piecewise_poisson"],
        default="piecewise_poisson",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--phases",
        type=str,
        default="10:0.2,10:1.0,10:0.2,10:1.0,10:0.2",
        help=(
            "Comma-separated duration_s:request_rate phases, for example "
            "'10:0.2,10:1.0,10:0.2,10:1.0,10:0.2'. "
            "Used only with --arrival-mode piecewise_poisson."
        ),
    )

    parser.add_argument(
        "--configs",
        type=str,
        default="2048,16384",
        help=(
            "Comma-separated server configurations. Supported values are "
            "'default', integer MBT values such as '2048' or '16384', "
            "'ppas_qwen', and 'ppas_nemotron'. Examples: "
            "'--configs 2048,16384' or "
            "'--configs ppas_qwen,768,2048'."
        ),
    )

    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="max_num_seqs used for all selected server configurations.",
    )

    return parser.parse_args()


def parse_phases(value: str) -> list[tuple[float, float]]:
    phases: list[tuple[float, float]] = []

    for item in value.split(","):
        item = item.strip()
        if not item:
            continue

        try:
            duration_s_str, rate_str = item.split(":", maxsplit=1)
            duration_s = float(duration_s_str)
            request_rate = float(rate_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid phase {item!r}. Expected duration_s:request_rate."
            ) from exc

        if duration_s <= 0:
            raise ValueError(f"Phase duration must be positive: {item!r}")
        if request_rate < 0:
            raise ValueError(f"Request rate must be non-negative: {item!r}")

        phases.append((duration_s, request_rate))

    if not phases:
        raise ValueError("At least one phase is required.")

    return phases


@dataclass(frozen=True)
class ServerConfig:
    name: str
    max_num_batched_tokens: int | None
    max_num_seqs: int | None
    max_num_scheduled_tokens: int | None = None
    ppas_enabled: bool = False
    ppas_b_cap: int | None = None

def parse_configs(
    value: str,
    max_num_seqs: int,
) -> list[ServerConfig]:
    configs: list[ServerConfig] = []
    seen: set[str] = set()

    for raw_item in value.split(","):
        item = raw_item.strip().lower()

        if not item:
            continue

        if item in seen:
            raise ValueError(
                f"Duplicate configuration in --configs: {item!r}"
            )
        seen.add(item)

        if item == "default":
            configs.append(
                ServerConfig(
                    name="default",
                    max_num_batched_tokens=None,
                    max_num_seqs=None,
                )
            )
            continue

        if item == "ppas_qwen":
            configs.append(
                ServerConfig(
                    name=f"ppas_2k_768_s{max_num_seqs}",
                    max_num_batched_tokens=2048,
                    max_num_seqs=max_num_seqs,
                    ppas_enabled=True,
                    ppas_b_cap=768,
                )
            )
            continue

        if item == "ppas_nemotron":
            configs.append(
                ServerConfig(
                    name=f"ppas_16k_1280_s{max_num_seqs}",
                    max_num_batched_tokens=16384,
                    max_num_seqs=max_num_seqs,
                    ppas_enabled=True,
                    ppas_b_cap=1280,
                )
            )
            continue

        try:
            mbt = int(item)
        except ValueError as exc:
            raise ValueError(
                f"Invalid configuration {raw_item!r}. "
                "Expected 'default', 'ppas_qwen', 'ppas_nemotron', or an integer MBT value."
            ) from exc

        if mbt <= 0:
            raise ValueError(
                f"max_num_batched_tokens must be positive, got {mbt}."
            )

        configs.append(
            ServerConfig(
                name=f"mbt_{mbt}_s{max_num_seqs}",
                max_num_batched_tokens=mbt,
                max_num_seqs=max_num_seqs,
            )
        )

    if not configs:
        raise ValueError("At least one configuration is required in --configs.")

    return configs


SERVER_SHUTDOWN_SLEEP_S = 3.0

@dataclass
class RequestSpec:
    arrival_s: float
    prompt_tokens: int
    output_tokens: int
    request_id: int
    phase_index: int = 0
    phase_rate: float = 0.0


def percentile(xs, p):
    if not xs:
        return float("nan")
    return float(np.percentile(xs, p))


def make_prompt(
    tokenizer: PreTrainedTokenizerBase,
    valid_token_ids: list[int],
    prompt_tokens: int,
    trace_seed: int,
    request_id: int,
) -> list[int]:
    rng = random.Random(trace_seed * 1_000_000 + request_id)

    prefix_length = min(128, prompt_tokens)
    unique_prefix = rng.choices(valid_token_ids, k=prefix_length)

    hello_ids = tokenizer.encode(
        " hello",
        add_special_tokens=False,
    )

    if len(hello_ids) != 1:
        raise RuntimeError(
            f"' hello' produced {len(hello_ids)} tokens instead of one."
        )

    filler_id = hello_ids[0]

    return unique_prefix + [filler_id] * (prompt_tokens - prefix_length)


def sample_trace(
    args: argparse.Namespace,
    phases: list[tuple[float, float]],
    trace_seed: int,
    duration_s: float,
) -> list[RequestSpec]:

    requests: list[RequestSpec] = []

    rng = random.Random(trace_seed)
    request_id = 0

    if args.arrival_mode == "piecewise_poisson":
        phase_start_s = 0.0

        for phase_index, (phase_duration_s, phase_rate) in enumerate(phases):
            phase_end_s = phase_start_s + phase_duration_s
            arrival_s = phase_start_s

            if phase_rate > 0:
                while True:
                    arrival_s += rng.expovariate(phase_rate)

                    if arrival_s >= phase_end_s:
                        break

                    requests.append(
                        RequestSpec(
                            arrival_s=arrival_s,
                            prompt_tokens=args.prompt_tokens,
                            output_tokens=args.output_tokens,
                            request_id=request_id,
                            phase_index=phase_index,
                            phase_rate=phase_rate,
                        )
                    )
                    request_id += 1

            phase_start_s = phase_end_s

    elif args.arrival_mode == "deterministic":
        num_requests = int(round(args.request_rate * duration_s))

        for request_id in range(num_requests):
            arrival_s = request_id / args.request_rate

            if arrival_s >= duration_s:
                break

            requests.append(
                RequestSpec(
                    arrival_s=arrival_s,
                    prompt_tokens=args.prompt_tokens,
                    output_tokens=args.output_tokens,
                    request_id=request_id,
                    phase_index=0,
                    phase_rate=args.request_rate,
                )
            )

    return requests

def trace_features(requests: list[RequestSpec], duration_s: float) -> dict:
    prompt_tokens = [r.prompt_tokens for r in requests]
    output_tokens = [r.output_tokens for r in requests]

    # approximate peak concurrency based only on requested output length is hard;
    # for v1 use arrival density proxy: max arrivals in any 1s window
    arrivals = [r.arrival_s for r in requests]
    peak_arrivals_1s = 0
    for t in arrivals:
        peak_arrivals_1s = max(peak_arrivals_1s, sum(1 for a in arrivals if t <= a < t + 1.0))

    return {
        "num_requests": len(requests),
        "request_rate": len(requests) / duration_s,
        "prompt_mean": mean(prompt_tokens) if prompt_tokens else 0,
        "prompt_p50": percentile(prompt_tokens, 50),
        "prompt_p95": percentile(prompt_tokens, 95),
        "prompt_max": max(prompt_tokens) if prompt_tokens else 0,
        "output_mean": mean(output_tokens) if output_tokens else 0,
        "output_p50": percentile(output_tokens, 50),
        "output_p95": percentile(output_tokens, 95),
        "output_max": max(output_tokens) if output_tokens else 0,
        "peak_arrivals_1s": peak_arrivals_1s,
    }


def start_vllm(
    args: argparse.Namespace,
    model: str,
    model_name: str,
    trace_id: int,
    config: ServerConfig,
) -> subprocess.Popen:
    vllm_cmd = [
        "vllm", "serve", model,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--no-enable-log-requests",
        "--max-model-len", "21000",
        "--language-model-only",
        "--skip-mm-profiling",
        "--kv-cache-dtype", "fp8_e4m3",
    ]

    if config.max_num_batched_tokens is not None:
        vllm_cmd += [
            "--max-num-batched-tokens",
            str(config.max_num_batched_tokens),
        ]
    if config.max_num_scheduled_tokens is not None:
        vllm_cmd += [
            "--max-num-scheduled-tokens",
            str(config.max_num_scheduled_tokens),
        ]
    if config.max_num_seqs is not None:
        vllm_cmd += [
            "--max-num-seqs",
            str(config.max_num_seqs),
        ]

    if args.profile:
        args.profile_dir.mkdir(parents=True, exist_ok=True)

        output_path = (
            args.profile_dir
            / f"{model_name}_trace_{trace_id}_{config.name}"
        )

        cmd = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--force-overwrite=true",
            "-o",
            str(output_path),
            *vllm_cmd,
        ]
    else:
        cmd = vllm_cmd

    env = os.environ.copy()
    env["PPAS_ENABLED"] = "1" if config.ppas_enabled else "0"

    if config.ppas_b_cap is not None:
        env["PPAS_B_CAP"] = str(config.ppas_b_cap)
    else:
        env.pop("PPAS_B_CAP", None)

    print(
        "START SERVER:",
        " ".join(cmd),
        f"PPAS_ENABLED={env['PPAS_ENABLED']}",
    )

    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )

def stop_vllm(proc: subprocess.Popen) -> None:
    print("STOP SERVER")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    time.sleep(SERVER_SHUTDOWN_SLEEP_S)


async def one_request(
    client: AsyncOpenAI,
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    valid_token_ids: list[int],
    spec: RequestSpec,
    trace_seed: int,
    trace_start: float,
) -> dict:
    prompt = make_prompt(
        tokenizer=tokenizer,
        valid_token_ids=valid_token_ids,
        prompt_tokens=spec.prompt_tokens,
        trace_seed=trace_seed,
        request_id=spec.request_id,
    )

    # Actual time at which this coroutine begins submitting the request.
    submitted_at = time.perf_counter()

    first_token_at = None
    last_token_at = None
    usage = None
    chunks = 0

    stream = await client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=spec.output_tokens,
        temperature=0.0,
        stream=True,
        extra_body={
            "ignore_eos": True,
            "stream_options": {"include_usage": True},
        },
    )

    async for chunk in stream:
        now = time.perf_counter()

        if chunk.choices:
            text = chunk.choices[0].text

            if text:
                if first_token_at is None:
                    first_token_at = now

                last_token_at = now
                chunks += 1

        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage

    completed_at = time.perf_counter()

    completion_tokens = (
        usage.completion_tokens if usage else spec.output_tokens
    )
    prompt_tokens = usage.prompt_tokens if usage else None

    latency = completed_at - submitted_at

    ttft = (
        None
        if first_token_at is None
        else first_token_at - submitted_at
    )

    tpot = None
    if first_token_at is not None:
        tpot = (
            completed_at - first_token_at
        ) / max(completion_tokens - 1, 1)

    return {
        "request_id": spec.request_id,

        # Planned and actual positions in the trace.
        "scheduled_arrival_s": spec.arrival_s,
        "submitted_s": submitted_at - trace_start,
        "submission_delay_s": (
            submitted_at - trace_start - spec.arrival_s
        ),
        "first_token_s": (
            None
            if first_token_at is None
            else first_token_at - trace_start
        ),
        "last_token_s": (
            None
            if last_token_at is None
            else last_token_at - trace_start
        ),
        "completed_s": completed_at - trace_start,

        # Existing duration metrics.
        "latency": latency,
        "ttft": ttft,
        "tpot": tpot,

        # Requested workload.
        "requested_prompt_tokens": spec.prompt_tokens,
        "requested_output_tokens": spec.output_tokens,
        "phase_index": spec.phase_index,
        "phase_rate": spec.phase_rate,

        # Actual API-reported usage.
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "chunks": chunks,
    }


async def replay_trace(
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    valid_token_ids: list[int],
    trace_seed: int,
    requests: list[RequestSpec],
) -> list[dict]:
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key="EMPTY",
    )

    trace_start = time.perf_counter()
    tasks = []

    async def schedule_request(spec: RequestSpec) -> dict:
        target_time = trace_start + spec.arrival_s
        sleep_s = target_time - time.perf_counter()

        if sleep_s > 0:
            await asyncio.sleep(sleep_s)

        return await one_request(
            client=client,
            model=model,
            tokenizer=tokenizer,
            valid_token_ids=valid_token_ids,
            spec=spec,
            trace_seed=trace_seed,
            trace_start=trace_start,
        )

    for spec in requests:
        tasks.append(
            asyncio.create_task(schedule_request(spec))
        )

    if not tasks:
        return []

    return await asyncio.gather(*tasks)

def summarize_run(results: list[dict]) -> dict:
    if not results:
        return {
            "avg_latency": float("nan"),
            "p95_latency": float("nan"),
            "avg_ttft": float("nan"),
            "p95_ttft": float("nan"),
            "avg_tpot": float("nan"),
            "p95_tpot": float("nan"),
        }

    latencies = [r["latency"] for r in results]
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    tpots = [r["tpot"] for r in results if r["tpot"] is not None]

    first_submission = min(r["submitted_s"] for r in results)
    last_completion = max(r["completed_s"] for r in results)

    makespan = last_completion - first_submission

    return {
        "avg_latency": mean(latencies),
        "p95_latency": percentile(latencies, 95),
        "avg_ttft": mean(ttfts),
        "p95_ttft": percentile(ttfts, 95),
        "avg_tpot": mean(tpots),
        "p95_tpot": percentile(tpots, 95),
        "makespan": makespan,
    }


def wait_for_server(
    proc: subprocess.Popen | None = None,
    timeout_s: float = 600,
) -> None:
    import requests

    start = time.time()
    last_error = None

    while time.time() - start < timeout_s:
        # Check process failure only when this script launched the server.
        if proc is not None and proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=5)

            stdout_text = (
                stdout.decode(errors="replace")
                if isinstance(stdout, bytes)
                else stdout or ""
            )
            stderr_text = (
                stderr.decode(errors="replace")
                if isinstance(stderr, bytes)
                else stderr or ""
            )

            raise RuntimeError(
                "vLLM process exited during startup\n\n"
                f"STDOUT:\n{stdout_text}\n\n"
                f"STDERR:\n{stderr_text}"
            )

        try:
            response = requests.get(
                f"{BASE_URL}/models",
                timeout=2,
            )
            if response.status_code == 200:
                return

            last_error = (
                f"status={response.status_code}, "
                f"body={response.text[:500]}"
            )
        except Exception as exc:
            last_error = repr(exc)

        time.sleep(2)

    # Only terminate a process that this script owns.
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait()

    raise RuntimeError(
        f"vLLM did not become ready after {timeout_s}s. "
        f"Last error: {last_error}"
    )


async def warmup_server(model: str) -> None:
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key="EMPTY",
    )

    for _ in range(3):
        await client.completions.create(
            model=model,
            prompt=("hello " * 128).strip(),
            max_tokens=32,
            temperature=0.0,
            extra_body={"ignore_eos": True},
        )

async def run_one(
    args: argparse.Namespace,
    model: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    valid_token_ids: list[int],
    trace_duration_s: float,
    trace_id: int,
    trace_seed: int,
    requests: list[RequestSpec],
    config: ServerConfig,
) -> None:
    features = trace_features(requests, trace_duration_s)

    proc: subprocess.Popen | None = None

    if args.reuse_server:
        print(f"REUSE SERVER: {BASE_URL}")
        print("Checking existing vLLM server...")
        wait_for_server(proc=None, timeout_s=30)
    else:
        proc = start_vllm(
            args=args,
            model=model,
            model_name=model_name,
            trace_id=trace_id,
            config=config,
        )
        print("Waiting for vLLM startup...")
        wait_for_server(proc=proc, timeout_s=600)

    if not args.skip_warmup:
        await warmup_server(model)

    failed = 0
    error = ""

    try:
        results = await replay_trace(
            model=model,
            tokenizer=tokenizer,
            valid_token_ids=valid_token_ids,
            trace_seed=trace_seed,
            requests=requests,
        )
        metrics = summarize_run(results)

        request_rows = [
            {
                "trace_id": trace_id,
                "configured_prompt_tokens": args.prompt_tokens,
                "configured_output_tokens": args.output_tokens,
                "configured_request_rate": (
                    "piecewise"
                    if args.arrival_mode == "piecewise_poisson"
                    else args.request_rate
                ),
                "phase_schedule": (
                    args.phases
                    if args.arrival_mode == "piecewise_poisson"
                    else ""
                ),
                "arrival_mode": args.arrival_mode,
                "seed": trace_seed,
                "config_name": config.name,
                "max_num_batched_tokens": (
                    config.max_num_batched_tokens
                    if config.max_num_batched_tokens is not None
                    else "default"
                ),
                "max_num_seqs": (
                    config.max_num_seqs
                    if config.max_num_seqs is not None
                    else "default"
                ),
                **result,
            }
            for result in results
        ]

        # workaround to make Nsight capture the end of the interesting workload
        if args.profile:
            await asyncio.sleep(1.0)
            for _ in range(10):
                await warmup_server(model)

    except Exception as e:
        failed = 1
        error = repr(e)
        metrics = {
            "avg_latency": float("nan"),
            "p95_latency": float("nan"),
            "avg_ttft": float("nan"),
            "p95_ttft": float("nan"),
            "avg_tpot": float("nan"),
            "p95_tpot": float("nan"),
        }
    finally:
        if proc is not None:
            stop_vllm(proc)


    row = {
        "trace_id": trace_id,
        "configured_prompt_tokens": args.prompt_tokens,
        "configured_output_tokens": args.output_tokens,
        "configured_request_rate": (
            "piecewise"
            if args.arrival_mode == "piecewise_poisson"
            else args.request_rate
        ),
        "phase_schedule": (
            args.phases
            if args.arrival_mode == "piecewise_poisson"
            else ""
        ),
        "arrival_mode": args.arrival_mode,
        "seed": trace_seed,
        **features,
        "config_name": config.name,
        "max_num_batched_tokens": (
            config.max_num_batched_tokens
            if config.max_num_batched_tokens is not None
            else "default"
        ),
        "max_num_seqs": (
            config.max_num_seqs
            if config.max_num_seqs is not None
            else "default"
        ),
        **metrics,
        "failed": failed,
        "error": error,
    }

    print(row)

async def main() -> None:
    args = parse_args()

    phases = parse_phases(args.phases)
    trace_seed = args.seed
    trace_id = 0

    model_name = args.model
    model = MODELS[model_name]

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    special_token_ids = set(tokenizer.all_special_ids)

    valid_token_ids = [
        token_id
        for token_id in range(len(tokenizer))
        if token_id not in special_token_ids
    ]

    trace_duration_s = (
        sum(duration_s for duration_s, _ in phases)
        if args.arrival_mode == "piecewise_poisson"
        else args.duration
    )

    configs = parse_configs(
        value=args.configs,
        max_num_seqs=args.max_num_seqs,
    )

    print("Selected server configurations:")
    for config in configs:
        print(
            f"  {config.name}: "
            f"max_num_batched_tokens={config.max_num_batched_tokens}, "
            f"max_num_scheduled_tokens={config.max_num_scheduled_tokens}, "
            f"max_num_seqs={config.max_num_seqs}, "
            f"ppas_enabled={config.ppas_enabled}, "
            f"ppas_b_cap={config.ppas_b_cap}"
        )
    requests = sample_trace(
        args=args,
        phases=phases,
        trace_seed=trace_seed,
        duration_s=trace_duration_s,
    )

    print("=" * 80)
    print("=" * 80)
    print(f"TRACE {trace_id}")
    print(f"arrival mode    : {args.arrival_mode}")

    if args.arrival_mode == "piecewise_poisson":
        print(f"phases          : {args.phases}")
    else:
        print(f"request rate    : {args.request_rate}")

    print(f"prompt tokens   : {args.prompt_tokens}")
    print(f"output tokens   : {args.output_tokens}")
    print(f"duration        : {trace_duration_s}")
    print(f"seed            : {trace_seed}")
    print(f"num requests    : {len(requests)}")
    print("=" * 80)

    if args.reuse_server:
        external_config = ServerConfig(
            name="external_server",
            max_num_batched_tokens=None,
            max_num_seqs=None,
            ppas_enabled=False,
        )

        await run_one(
            args=args,
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            valid_token_ids=valid_token_ids,
            trace_duration_s=trace_duration_s,
            trace_id=trace_id,
            trace_seed=trace_seed,
            requests=requests,
            config=external_config,
        )

    else:
        for config in configs:
            await run_one(
                args=args,
                model=model,
                model_name=model_name,
                tokenizer=tokenizer,
                valid_token_ids=valid_token_ids,
                trace_duration_s=trace_duration_s,
                trace_id=trace_id,
                trace_seed=trace_seed,
                requests=requests,
                config=config,
            )


if __name__ == "__main__":
    asyncio.run(main())
