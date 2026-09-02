"""
Inspect all recordings after preprocessing.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from src.io.loader import DatasetLoader

from src.preprocessing.presence import detect_presence
from src.preprocessing.trim import trim

from src.visualization.plot import (
    plot_scene,
    plot_x_time,
    plot_speed,
    plot_raw_vs_solved,
)

OUTPUT = Path("results/inspection")
OUTPUT.mkdir(parents=True, exist_ok=True)

raw_loader = DatasetLoader(
    root="data/OptiTrack/edited_data/raw",
    config="config/recordings.yaml",
)

solved_loader = DatasetLoader(
    root="data/OptiTrack/edited_data/solved",
    config="config/recordings.yaml",
)

fps = raw_loader.config["dataset"]["fps"]
waiting_area_x = raw_loader.config["preprocessing"]["waiting_area_x"]

for session_name, session_cfg in raw_loader.config["sessions"].items():

    session = int(session_name.split("_")[1])

    for scenario in session_cfg["scenarios"]:

        print(f"Session {session}  Scenario {scenario}")

        _, raw_df, info = raw_loader.load(
            session=session,
            scenario=scenario,
        )

       
        _, solved_df, _ = solved_loader.load(
            session=session,
            scenario=scenario,
        )

        presence = detect_presence(
            df=raw_df,
            bodies=info["humans"] + info["robots"],
            waiting_area_x=waiting_area_x,
        )

        solved_df = trim(
            solved_df,
            presence,
        )
        print(solved_df
              .head())

        output_dir = (
            OUTPUT /
            f"session_{session}" /
            f"scenario_{scenario}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        
        
        fig, _ = plot_scene(
            solved_df,
            info,
        )

        fig.savefig(
            output_dir / "scene.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

   

        fig, _ = plot_x_time(
            solved_df,
            info,
        )

        fig.savefig(
            output_dir / "x_time.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

       
        fig, _ = plot_speed(
            solved_df,
            info,
            fps,
        )

        fig.savefig(
            output_dir / "speed.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        
        fig, _ = plot_raw_vs_solved(
            raw_df,
            solved_df,
            info,
        )

        fig.savefig(
            output_dir / "raw_vs_solved.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

print("\nInspection finished.")
print(f"Results saved to: {OUTPUT.resolve()}")