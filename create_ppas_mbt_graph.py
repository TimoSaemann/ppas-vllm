#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

BURST = np.array([0.5, 1.0, 1.5, 2.0])

# LATENCY = {
#     "MBT 1280": [1.653, 2.323, 4.562, 5.771],
#     "MBT 2048": [1.546, 2.418, 4.899, 6.155],
#     "MBT 16k":  [1.454, 2.549, 6.637, 8.936],
#     "P-PAS":    [1.441, 2.292, 4.425, 5.578],
# }

LATENCY = {
    "MBT 1280": [1.556, 2.296, 4.462, 5.663],
    "MBT 2048": [1.507, 2.307, 4.740, 6.012],
    "MBT 16k":  [1.440, 2.566, 6.869, 9.971],
    "P-PAS":    [1.444, 2.333, 4.449, 5.588],
}


plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


fig, ax = plt.subplots(figsize=(7.6, 4.8))


# ---------------------------------------------------------------------
# Main curves
# ---------------------------------------------------------------------

for name in ["MBT 1280", "MBT 2048", "MBT 16k"]:
    ax.plot(
        BURST,
        LATENCY[name],
        "-x",
        linewidth=2.0,
        markersize=5,
        label=name,
        zorder=5,
    )

ax.plot(
    BURST,
    LATENCY["P-PAS"],
    "-o",
    linewidth=2.2,
    markersize=5,
    markerfacecolor="white",
    markeredgewidth=1.5,
    label="P-PAS",
    zorder=6,
)


# ---------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------

ax.set_xlabel("Burst arrival rate [requests/s]")
ax.set_ylabel("Average end-to-end latency [s]")

ax.set_xticks(BURST)

ax.grid(
    True,
    axis="y",
    alpha=0.25,
)

ax.legend(
    ncol=2,
    frameon=False,
    loc="upper left",
)


# ---------------------------------------------------------------------
# Zoom inset: burst rate 0.5
# ---------------------------------------------------------------------

axins = inset_axes(
    ax,
    width="43%",
    height="42%",
    loc="upper left",
    bbox_to_anchor=(0.05, 0.08, 1.0, 0.72),
    bbox_transform=ax.transAxes,
)

for name in ["MBT 1280", "MBT 2048", "MBT 16k"]:
    axins.plot(
        BURST,
        LATENCY[name],
        "-x",
        linewidth=1.5,
        markersize=5,
        zorder=4,
    )

axins.plot(
    BURST,
    LATENCY["P-PAS"],
    "-o",
    linewidth=1.8,
    markersize=5,
    markerfacecolor="white",
    markeredgewidth=1.3,
    zorder=6,
)

# Tight zoom around the 0.5 burst-rate results
axins.set_xlim(0.47, 0.53)
axins.set_ylim(1.40, 1.58)

axins.set_xticks([0.5])
axins.set_yticks([
    1.40,
    1.44,
    1.48,
    1.52,
    1.56,
])

axins.tick_params(labelsize=8)
axins.grid(
    True,
    axis="y",
    alpha=0.2,
)

mark_inset(
    ax,
    axins,
    loc1=2,
    loc2=4,
    fc="none",
    ec="0.5",
)


fig.tight_layout()

fig.savefig(
    OUT_DIR / "nemotron_latency_vs_burst.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT_DIR / "nemotron_latency_vs_burst.pdf",
    bbox_inches="tight",
)

plt.show()