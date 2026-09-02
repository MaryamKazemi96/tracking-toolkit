"""
Plot THÖR-style metrics for the complete dataset.

For each metric, compare:
    - Overall
    - Low-density
    - High-density

Both mean ± standard deviation and median are plotted.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np



INPUT = Path(
    "results/thor_metrics/dataset_metrics.json"
)

OUTPUT = Path(
    "results/thor_metrics/plots"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)



# Metrics


METRICS = {
    "tracking_duration": {
        "label": "Tracking duration",
        "unit": "s",
    },
    "trajectory_curvature": {
        "label": "Trajectory curvature",
        "unit": r"m$^{-1}$",
    },
    "perception_noise_savgol": {
        "label": "Perception noise",
        "unit": r"m/s$^2$",
    },
    "motion_speed": {
        "label": "Motion speed",
        "unit": "m/s",
    },
    "minimum_distance_between_people": {
        "label": "Minimum distance between people",
        "unit": "m",
    },
}


GROUPS = [
    ("overall", "Overall"),
    ("low_density", "Low density"),
    ("high_density", "High density"),
]



# Load data


with open(
    INPUT,
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)




def get_metric_values(metric_name: str):
    """
    Return mean, std, and median for all density groups.
    """

    means = []
    stds = []
    medians = []

    for group_key, _ in GROUPS:

        result = data[group_key][metric_name]

        means.append(result["mean"])
        stds.append(result["std"])
        medians.append(result["median"])

    return (
        np.asarray(means, dtype=float),
        np.asarray(stds, dtype=float),
        np.asarray(medians, dtype=float),
    )


def save_figure(fig, name: str):

    fig.savefig(
        OUTPUT / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT / f"{name}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

for metric_name, config in METRICS.items():

    means, stds, medians = get_metric_values(
        metric_name
    )

    labels = [
        label
        for _, label in GROUPS
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(7, 4.8)
    )

    ax.errorbar(
        x - width / 2,
        means,
        yerr=stds,
        fmt="o",
        capsize=5,
        markersize=7,
        label="Mean ± std",
    )

  
    ax.scatter(
        x + width / 2,
        medians,
        marker="s",
        s=55,
        label="Median",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(
        f"{config['label']} [{config['unit']}]"
    )

    ax.set_title(
        config["label"]
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    save_figure(
        fig,
        f"{metric_name}_mean_median",
    )


labels = [
    label
    for _, label in GROUPS
]

x = np.arange(
    len(labels)
)

fig, axes = plt.subplots(
    5,
    1,
    figsize=(7, 18),
)

for ax, (metric_name, config) in zip(
    axes,
    METRICS.items(),
):

    means, stds, _ = get_metric_values(
        metric_name
    )

    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o",
        capsize=5,
        markersize=6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(
        f"{config['label']} [{config['unit']}]"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

fig.suptitle(
    "THÖR-style metrics — Mean ± Standard Deviation",
    fontsize=14,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.98]
)

save_figure(
    fig,
    "thor_metrics_mean_std",
)

fig, axes = plt.subplots(
    5,
    1,
    figsize=(7, 18),
)

for ax, (metric_name, config) in zip(
    axes,
    METRICS.items(),
):

    _, _, medians = get_metric_values(
        metric_name
    )

    ax.bar(
        x,
        medians,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(
        f"{config['label']} [{config['unit']}]"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

fig.suptitle(
    "THÖR-style metrics — Median",
    fontsize=14,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.98]
)

save_figure(
    fig,
    "thor_metrics_median",
)



fig, axes = plt.subplots(
    5,
    1,
    figsize=(7, 18),
)

for ax, (metric_name, config) in zip(
    axes,
    METRICS.items(),
):

    means, stds, medians = get_metric_values(
        metric_name
    )

    width = 0.35


    ax.errorbar(
        x - width / 2,
        means,
        yerr=stds,
        fmt="o",
        capsize=4,
        markersize=5,
        label="Mean ± std",
    )

    # Median
    ax.scatter(
        x + width / 2,
        medians,
        marker="s",
        s=45,
        label="Median",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(
        f"{config['label']} [{config['unit']}]"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.legend(
        loc="best"
    )

fig.suptitle(
    "THÖR-style metrics — Mean and Median",
    fontsize=14,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.98]
)

save_figure(
    fig,
    "thor_metrics_mean_median",
)



print(
    f"\nPlots saved to:\n{OUTPUT.resolve()}"
)