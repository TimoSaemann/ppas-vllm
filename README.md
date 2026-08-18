# P-PAS: Prefill-Pressure Adaptive Scheduling for Long-Context LLM Serving

This repository contains the code and experimental results for:

> **P-PAS: Prefill-Pressure Adaptive Scheduling for Long-Context LLM Serving**  
> Timo Sämann  
> [arXiv link](https://arxiv.org/abs/2608.15171)


![Motivation and overview of P-PAS. ](figures/overview_figure.png)


Long-context LLM applications such as retrieval-augmented generation (RAG) and agentic systems often process tens of thousands of input tokens to produce short outputs, making end-to-end request latency an important serving objective.
We show that the maximum number of batched tokens (MBT), which controls the
token scheduling budget in vLLM, has a scheduling-pressure-dependent effect
on latency. Larger token budgets can reduce latency under low scheduling
pressure, while smaller budgets become preferable under higher pressure.
Consequently, no single static MBT performs best across load regimes.

We introduce Prefill-Pressure Adaptive Scheduling (P-PAS), a lightweight
policy that dynamically adapts the scheduling budget based on concurrent
prefill and decode state. P-PAS retains a large token budget under low
pressure and constrains prefill work as pressure increases. Across models,
workloads, and GPUs, P-PAS maintains low end-to-end latency across changing
load regimes, avoiding the limitations of a fixed MBT.

Kernel-level profiling shows that large prefill chunks can improve execution
efficiency under low scheduling pressure, but that this advantage varies
across model–hardware configurations. As scheduling pressure increases,
smaller chunks can instead reduce interference with active decoding,
explaining the observed load-dependent MBT sensitivity.

## Repository Structure

```text
ppas-vllm/
├── benchmark.py
├── run_sweep.py
├── evaluate_results.py
├── scheduler/
│   ├── scheduler_ppas.py
│   └── ppas_vllm_0.22.1.patch
├── paper_results/
│   ├── ...
│   ├── nsys_systems/
│   └── nsys_compute/
├── LICENSE
└── README.md
```


- `benchmark.py` generates the long-context serving workloads.
- `run_sweep.py` runs experiments across scheduler configurations and seeds.
- `evaluate_results.py` aggregates the benchmark results.
- `scheduler/` contains the complete P-PAS scheduler and the patch relative to vLLM 0.22.1.
- `paper_results/` contains the raw experimental results and NVIDIA Nsight profiling data used in the paper.

## Installation and vLLM Integration

### Installation

Create and activate a Conda environment with Python 3.12:

```bash
conda create -n ppas-vllm python=3.12 -y
conda activate ppas-vllm
python -m pip install --upgrade pip setuptools wheel
```

Install the required packages:

```bash
pip install \
  vllm==0.22.1 \
  transformers==5.10.2 \
  openai==2.41.0 \
  numpy==2.3.5 \
  pandas==3.0.3 \
  huggingface_hub==1.18.0 \
  requests==2.34.2
```

Download the models used in the experiments from the Hugging Face Hub:

```bash
hf download Qwen/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-0.5B-Instruct
hf download HuggingFaceTB/SmolLM3-3B
```

### P-PAS Scheduler

P-PAS is implemented as a lightweight modification of the vLLM 0.22.1
scheduler (`vllm/v1/core/sched/scheduler.py`).

The repository provides both:

- `scheduler/ppas_vllm_0.22.1.patch`: patch against the vLLM 0.22.1 scheduler
- `scheduler/scheduler_ppas.py`: complete modified scheduler for inspection or direct replacement

The patch is the recommended installation method.

Locate the active Python environment and apply the patch:

```bash
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cd "$SITE_PACKAGES"
patch -p1 < /path/to/ppas-vllm/scheduler/ppas_vllm_0.22.1.patch
```

The patch is version-specific and targets vLLM 0.22.1.

The patched scheduler behaves as the standard vLLM scheduler unless
`PPAS_ENABLED=1` is set.

### Tested Environment

Experiments were conducted on the **NVIDIA GeForce RTX 5090**,
**NVIDIA A100-SXM4-80GB**, and **NVIDIA H100 80GB HBM3**.

The RTX 5090 and A100 experiments used:

```text
PyTorch:          2.11.0+cu130
PyTorch CUDA:     13.0
vLLM:             0.22.1
Transformers:     5.10.2
OpenAI:           2.41.0
NumPy:            2.3.5
Pandas:           3.0.3
Hugging Face Hub: 1.18.0
Requests:         2.34.2
```

The H100 experiments used the same software versions, but with
**PyTorch 2.11.0+cu128 and CUDA 12.8**.



## Running the Benchmarks

Use `benchmark.py` to run a single workload across one or more scheduler
configurations:

```bash
python benchmark.py \
  --model qwen_3b \
  --prompt-tokens 25000 \
  --output-tokens 32 \
  --arrival-mode piecewise_poisson \
  --phases 10:0.2,10:0.8,10:0.2,10:0.8,10:0.2 \
  --configs 2048,16384,ppas
```

Here, `2048` and `16384` select the fixed-MBT baselines, while `ppas`
enables P-PAS.

See all available options with:

```bash
python benchmark.py --help
```

Use `run_sweep.py` to automate experiments across multiple models, request
shapes, load regimes, and seeds:

```bash
python run_sweep.py --help
```

> **Benchmarking note:** For stable latency measurements, run the benchmark on
> an otherwise idle GPU. If the benchmark GPU also drives the display,
> GPU-accelerated desktop or browser activity can noticeably affect the
> measured latencies.

## Aggregating the Paper Results

The raw outputs from the experiments reported in the paper are included under
`paper_results/`.

The reported metrics can be computed from these outputs using, for example:

```bash
python evaluate_results.py --input /path/to/ppas-vllm/paper_results/RTX5090/RTX5090_qwen_3b.txt
```

This aggregates the existing benchmark results without rerunning the GPU
experiments.

## Profiling

### Nsight Systems

Nsight Systems profiling is integrated into `benchmark.py`. Add `--profile`
to run the vLLM server under Nsight Systems. For kernel-level analysis, we use
deterministic arrivals and a sufficiently low request rate to isolate clean
single-request execution; the exact rate depends on the model size.

```bash
python benchmark.py \
  --model qwen_3b \
  --prompt-tokens 25000 \
  --output-tokens 32 \
  --arrival-mode deterministic \
  --request-rate 0.1 \
  --configs 2048 \
  --profile \
  --profile-dir nsys_profiles
```

Nsight Systems traces are written to the directory specified by
`--profile-dir` (default: `nsys_profiles/`).

The generated trace can be opened with NVIDIA Nsight Systems.

### Nsight Compute

Nsight Compute profiling is performed by starting the vLLM server separately
under `ncu` and then running the benchmark against the existing server:

```bash
python benchmark.py \
  --model qwen_3b \
  --prompt-tokens 25000 \
  --output-tokens 32 \
  --arrival-mode deterministic \
  --request-rate 0.1 \
  --configs 2048 \
  --reuse-server \
  --skip-warmup
```

A representative `ncu` invocation is:

```bash
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:<kernel-name>' \
  --launch-count 1 \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --export profile \
  vllm serve <model> \
    --host 0.0.0.0 \
    --port 8000 \
    --no-enable-log-requests \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 256
```

The target kernel name can first be identified from an Nsight Systems trace.
Depending on the model and kernel launch order, `--launch-skip` may be required
to select the intended kernel invocation.

The Nsight Systems and Nsight Compute traces used for the paper are included
under `paper_results/` and can be inspected directly without rerunning the
profiling experiments.

## Citation

If you use P-PAS or this repository in your research, please cite:

```bibtex
@misc{sämann2026ppasprefillpressureadaptivescheduling,
      title={P-PAS: Prefill-Pressure Adaptive Scheduling for Long-Context LLM Serving}, 
      author={Timo Sämann},
      year={2026},
      eprint={2608.15171},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2608.15171}, 
}
```

## License

This repository is licensed under the Apache License 2.0. See
`LICENSE` for details.