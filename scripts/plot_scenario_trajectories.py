"""
Generate the "trajectories across the four scenarios" figures for the paper

"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.io.loader import DatasetLoader
from src.preprocessing.presence import detect_presence
from src.preprocessing.trim import trim
from src.visualization.plot import plot_scene, compute_global_xy_limits

# ---------------------------------------------------------------------

ROOT_SOLVED = "data/OptiTrack/solved"
ROOT_RAW = "data/OptiTrack/raw"
CONFIG = "config/recordings.yaml"

OUTPUT = Path("results/paper_figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

ARENA_MAP_PATH = "data/figures/env5.png"
ARENA_MAP_EXTENT = (-4.6, 6.282, -3.1920, 3.058)  # (xmin, xmax, ymin, ymax) in meters

ADDITIONAL_OBSTACLE_REGIONS = [
    (-1, 2, -2, 2),
]


def load_trimmed(solved_loader, raw_loader, session, scenario):
    """Load a scenario's solved trajectories, trimmed to presence."""

    _, raw_df, info = raw_loader.load(session=session, scenario=scenario)
    _, solved_df, _ = solved_loader.load(session=session, scenario=scenario)

    waiting_area_x = raw_loader.config["preprocessing"]["waiting_area_x"]

    presence = detect_presence(
        df=raw_df,
        bodies=info["humans"] + info["robots"],
        waiting_area_x=waiting_area_x,
    )

    solved_df = trim(solved_df, presence)

    return solved_df, info


def plot_session_scenarios(solved_loader, raw_loader, session, scenario_ids):
    """1x4 panel of Scenario 1-4 trajectories for one session, same shape/limits."""

    
    loaded = [
        (scenario, *load_trimmed(solved_loader, raw_loader, session, scenario))
        for scenario in scenario_ids
    ]

    dfs_infos = [(df, info) for _, df, info in loaded]

    if ARENA_MAP_EXTENT is not None:
        print(f"Using ARENA_MAP_EXTENT {ARENA_MAP_EXTENT} for all panels.")
       
        xlim = (ARENA_MAP_EXTENT[0], ARENA_MAP_EXTENT[1])
        ylim = (ARENA_MAP_EXTENT[2], ARENA_MAP_EXTENT[3])
    else:
        xlim, ylim = compute_global_xy_limits(dfs_infos, padding=0.3)

    fig, axes = plt.subplots(1, len(scenario_ids), figsize=(5.2 * len(scenario_ids), 5.2))

    if len(scenario_ids) == 1:
        axes = [axes]

    for ax, (scenario, df, info) in zip(axes, loaded):

        plot_scene(
            df,
            info,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            background_image=ARENA_MAP_PATH,
            background_extent=ARENA_MAP_EXTENT,
            mask_regions=ADDITIONAL_OBSTACLE_REGIONS if scenario == 1 else None,
        )

        ax.set_title(f"Scenario {scenario}")
        ax.legend().set_visible(False)  # avoid 4x repeated legends cluttering the figure

    handles, labels = axes[-1].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=8,
    )

    fig.suptitle(f"Session {session} — trajectories across scenarios", y=1.02)
    fig.tight_layout()

    return fig


def main() -> None:

    solved_loader = DatasetLoader(root=ROOT_SOLVED, config=CONFIG)
    raw_loader = DatasetLoader(root=ROOT_RAW, config=CONFIG)

    density_examples = {}  

    for session_name, session_cfg in solved_loader.config["sessions"].items():

        session = int(session_name.split("_")[1])
        scenario_ids = sorted(session_cfg["scenarios"])
        density = session_cfg["density"]

        density_examples.setdefault(density, session)

        print(f"Plotting Session {session} ({density} density) ...")

        fig = plot_session_scenarios(solved_loader, raw_loader, session, scenario_ids)

        fig.savefig(OUTPUT / f"session_{session}_scenarios.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUTPUT / f"session_{session}_scenarios.pdf", bbox_inches="tight")
        plt.close(fig)

   
    if "low" in density_examples and "high" in density_examples:

        loaded = {
            density: load_trimmed(solved_loader, raw_loader, density_examples[density], 4)
            for density in ["low", "high"]
        }

        if ARENA_MAP_EXTENT is not None:
            xlim = (ARENA_MAP_EXTENT[0], ARENA_MAP_EXTENT[1])
            ylim = (ARENA_MAP_EXTENT[2], ARENA_MAP_EXTENT[3])
        else:
            xlim, ylim = compute_global_xy_limits(list(loaded.values()), padding=0.3)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

        for ax, density in zip(axes, ["low", "high"]):

            df, info = loaded[density]

            plot_scene(
                df,
                info,
                ax=ax,
                xlim=xlim,
                ylim=ylim,
                background_image=ARENA_MAP_PATH,
                background_extent=ARENA_MAP_EXTENT,
            )

            ax.set_title(f"{density.capitalize()} density (Session {density_examples[density]}, Scenario 4)")

        fig.suptitle("Low- vs high-density crowd trajectories", y=1.03)
        fig.tight_layout()

        fig.savefig(OUTPUT / "density_comparison.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUTPUT / "density_comparison.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"\nDone. Figures saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()