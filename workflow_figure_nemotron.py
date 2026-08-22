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
    (10, 2.0),
    (10, 0.1),
    (10, 2.0),
    (10, 0.1),
]

BURST_THRESHOLD = 0.5
SEED = 3

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

# upper broken-axis section / lower section / arrival raster
gs = fig.add_gridspec(
    nrows=3,
    ncols=1,
    height_ratios=[1.25, 3.7, 0.7],
    hspace=0.04,
)

ax_hi = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1], sharex=ax_hi)
ax_raster = fig.add_subplot(gs[2], sharex=ax_hi)

# ---------------------------------------------------------------------
# Shade burst phases
# ---------------------------------------------------------------------

BURST_COLOR = "tab:blue"

start = 0.0

for duration, rate in PHASES:
    end = start + duration
    midpoint = (start + end) / 2

    if rate >= BURST_THRESHOLD:

        ax.axvspan(
            start,
            end,
            ymin=(0.5 - 0.0) / (0.6 - 0.0),
            ymax=1.0,
            alpha=0.10,
            color=BURST_COLOR,
        )

        ax_hi.fill_between(
            [start, end],
            1.7,
            rate,
            alpha=0.10,
            color=BURST_COLOR,
        )

        ax_raster.axvspan(
            start,
            end,
            alpha=0.10,
            color=BURST_COLOR,
        )

        ax_hi.text(
            midpoint,
            2.13,
            "Burst\n" + rf"$\lambda = {rate:.1f}$",
            ha="center",
            va="center",
            fontsize=15,
        )

    else:
        ax.text(
            midpoint,
            0.16,
            "Steady load\n" + rf"$\lambda = {rate:.1f}$",
            ha="center",
            va="center",
            fontsize=14,
        )

    start = end

# ---------------------------------------------------------------------
# Arrival-rate trace on both broken axes
# ---------------------------------------------------------------------

for axis in (ax_hi, ax):
    axis.step(
        times,
        rates,
        where="post",
        linewidth=2.8,
        label=r"Arrival rate $\lambda(t)$",
        zorder=3,
    )

# ---------------------------------------------------------------------
# Broken y-axis limits
# ---------------------------------------------------------------------

# Lower section shows steady-load region and shading start.
ax.set_ylim(0.0, 0.6)

# Upper section shows burst region around lambda=2.
ax_hi.set_ylim(1.7, 2.15)

ax.set_yticks([
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
])

ax_hi.set_yticks([
    1.8,
    2.0,
])

# Hide touching spines.
ax_hi.spines["bottom"].set_visible(False)
ax.spines["top"].set_visible(False)

ax_hi.tick_params(
    axis="x",
    bottom=False,
    labelbottom=False,
)

ax.tick_params(
    axis="x",
    bottom=False,
    labelbottom=False,
)

# ---------------------------------------------------------------------
# Diagonal break marks
# ---------------------------------------------------------------------

d = 0.015

kwargs = dict(
    transform=ax_hi.transAxes,
    clip_on=False,
    linewidth=1.5,
)

ax_hi.plot(
    (-d, +d),
    (-d, +d),
    **kwargs,
)

kwargs["transform"] = ax.transAxes

ax.plot(
    (-d, +d),
    (1 - d, 1 + d),
    **kwargs,
)

# ---------------------------------------------------------------------
# Common formatting
# ---------------------------------------------------------------------

ax.set_xlim(
    0,
    total_duration,
)

ax.set_ylabel(
    "Arrival rate [requests/s]"
)

ax.grid(
    axis="y",
    alpha=0.22,
)

ax_hi.grid(
    axis="y",
    alpha=0.22,
)

ax_hi.legend(
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(0.0, 1.01),
    borderaxespad=0.0,
)

for axis in (ax_hi, ax):
    axis.spines["right"].set_visible(False)

ax_hi.spines["top"].set_visible(False)

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
    zorder=4,
)

ax_raster.text(
    -0.8,
    0.42,
    "Poisson arrivals",
    ha="right",
    va="center",
    fontsize=14,
)

ax_raster.set_ylim(
    0,
    1,
)

ax_raster.set_xlabel(
    "Time [s]"
)

ax_raster.set_xticks(
    np.arange(
        0,
        total_duration + 1,
        10,
    )
)

ax_raster.set_yticks([])

ax_raster.spines["top"].set_visible(False)
ax_raster.spines["right"].set_visible(False)
ax_raster.spines["left"].set_visible(False)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

fig.savefig(
    OUT_DIR / "nemotron_piecewise_poisson_workload.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.08,
)

plt.show()