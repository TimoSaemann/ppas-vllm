#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Workload definition
# ---------------------------------------------------------------------

PHASES = [
    (10, 0.1),
    (10, 0.8),
    (10, 0.1),
    (10, 0.8),
    (10, 0.1),
]

SEED = 1

# ---------------------------------------------------------------------
# Generate piecewise-Poisson arrivals
# ---------------------------------------------------------------------

rng = np.random.default_rng(SEED)

arrival_times = []
phase_start = 0.0

for duration, rate in PHASES:
    phase_end = phase_start + duration
    t = phase_start

    while True:
        t += rng.exponential(1.0 / rate)

        if t >= phase_end:
            break

        arrival_times.append(t)

    phase_start = phase_end

arrival_times = np.asarray(arrival_times)

# ---------------------------------------------------------------------
# Construct step-plot coordinates
# ---------------------------------------------------------------------

times = [0.0]
rates = []

current_time = 0.0

for duration, rate in PHASES:
    rates.append(rate)
    current_time += duration
    times.append(current_time)

rates.append(PHASES[-1][1])

times = np.asarray(times)
rates = np.asarray(rates)

total_duration = sum(duration for duration, _ in PHASES)

# ---------------------------------------------------------------------
# Plot setup
# ---------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

fig = plt.figure(
    figsize=(11.0, 5.4),
    constrained_layout=True,
)

gs = fig.add_gridspec(
    nrows=2,
    ncols=1,
    height_ratios=[4.0, 0.8],
    hspace=0.05,
)

ax = fig.add_subplot(gs[0])
ax_raster = fig.add_subplot(gs[1], sharex=ax)

# ---------------------------------------------------------------------
# Arrival-rate trace
# ---------------------------------------------------------------------

ax.step(
    times,
    rates,
    where="post",
    linewidth=2.8,
    label=r"Arrival rate $\lambda(t)$",
)

# ---------------------------------------------------------------------
# Phase labels
# ---------------------------------------------------------------------

start = 0.0

for duration, rate in PHASES:
    end = start + duration
    midpoint = (start + end) / 2

    if rate == 0.8:
        ax.text(
            midpoint,
            0.84,
            "Burst\n" + rf"$\lambda = {rate:.1f}$",
            ha="center",
            va="bottom",
            fontsize=15,
        )
    else:
        ax.text(
            midpoint,
            0.18,
            "Steady load\n" + rf"$\lambda = {rate:.1f}$",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    start = end

# ---------------------------------------------------------------------
# Common formatting
# ---------------------------------------------------------------------

ax.set_xlim(0, total_duration)
ax.set_ylim(0.0, 1.0)

ax.set_yticks([
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
])

ax.set_ylabel("Arrival rate [requests/s]")

ax.grid(axis="y", alpha=0.22)

ax.legend(
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(0.0, 1.01),
    borderaxespad=0.0,
)

ax.tick_params(axis="x", bottom=False, labelbottom=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ---------------------------------------------------------------------
# Poisson-arrival raster
# ---------------------------------------------------------------------

ax_raster.vlines(
    arrival_times,
    0.0,
    0.82,
    color="tab:orange",
    linewidth=2.0,
    alpha=0.95,
)

ax_raster.text(
    -0.8,
    0.42,
    "Poisson arrivals",
    ha="right",
    va="center",
    fontsize=14,
)

ax_raster.set_ylim(0, 1)
ax_raster.set_xlabel("Time [s]")

ax_raster.set_xticks(
    np.arange(0, total_duration + 1, 10)
)

ax_raster.set_yticks([])

ax_raster.spines["top"].set_visible(False)
ax_raster.spines["right"].set_visible(False)
ax_raster.spines["left"].set_visible(False)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

fig.savefig(
    OUT_DIR / "piecewise_poisson_workload.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.08,
)

plt.show()