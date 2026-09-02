"""
Compute THÖR-style metrics for the complete dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.io.loader import DatasetLoader
from src.preprocessing.presence import detect_presence
from src.preprocessing.trim import trim

from src.metrics.thor_metrics import (
    compute_recording_metrics,
    summarize_dataset_metrics,
)


OUTPUT = Path("results/thor_metrics")
OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


# Known corrupted recordings
EXCLUDED_RECORDINGS = {

}



loader = DatasetLoader(
    root="data/OptiTrack/solved",
    config="config/recordings.yaml",
)
loader_raw= DatasetLoader(
    root="data/OptiTrack/raw",
    config="config/recordings.yaml",
)
fps = loader.config["dataset"]["fps"]

waiting_area_x = (
    loader.config["preprocessing"]
    ["waiting_area_x"]
)


METRIC_NAMES = [
    "tracking_duration",
    "trajectory_curvature",
    "perception_noise_savgol",
    "motion_speed",
    "minimum_distance_between_people",
]


def empty_metrics():
    """Create an empty metric dictionary."""
    return {
        metric: []
        for metric in METRIC_NAMES
    }


all_metrics = empty_metrics()
density_metrics = {
    "low": empty_metrics(),
    "high": empty_metrics(),
}

for session_name, session_cfg in loader.config["sessions"].items():

    session = int(
        session_name.split("_")[1]
    )

    density = session_cfg["density"].lower()

    if density not in density_metrics:
        raise ValueError(
            f"Unknown density '{density}' for "
            f"session {session}. Expected 'low' or 'high'."
        )

    print(
        f"\nSession {session} "
        f"({density} density)"
    )

    for scenario in session_cfg["scenarios"]:

        if (session, scenario) in EXCLUDED_RECORDINGS:

            print(
                f"  Skipping corrupted recording: "
                f"Scenario {scenario}"
            )

            continue

        print(
            f"  Processing Scenario {scenario}"
        )

        _, df, info = loader.load(
            session=session,
            scenario=scenario,
        )
        _, df_raw, info_raw = loader_raw.load(
            session=session,
            scenario=scenario,
        )
        humans = info["humans"]
        robots = info["robots"]

        bodies = humans + robots

        # Detect presence

        presence = detect_presence(
            df=df_raw,
            bodies=bodies,
            waiting_area_x=waiting_area_x,
        )

        # trim
        df = trim(
            df,
            presence,
        )


        metrics = compute_recording_metrics(
            df=df,
            humans=humans,
            fps=fps,
            curvature_stride=1.0,
            smoothing_window_seconds=1.0,
            smoothing_polynomial_order=3,
        )


        for metric_name in METRIC_NAMES:

            values = metrics.get(
                metric_name,
                [],
            )

            all_metrics[metric_name].extend(
                values
            )


            density_metrics[density][
                metric_name
            ].extend(values)


        recording_summary = (
            summarize_dataset_metrics(
                metrics
            )
        )

        output_file = (
            OUTPUT
            / f"session_{session}"
            / f"scenario_{scenario}.json"
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                recording_summary,
                f,
                indent=2,
            )


overall_summary = summarize_dataset_metrics(
    all_metrics
)

low_density_summary = summarize_dataset_metrics(
    density_metrics["low"]
)

high_density_summary = summarize_dataset_metrics(
    density_metrics["high"]
)


dataset_summary = {
    "overall": overall_summary,
    "low_density": low_density_summary,
    "high_density": high_density_summary,
}


output_file = (
    OUTPUT / "dataset_metrics.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        dataset_summary,
        f,
        indent=2,
    )


METRICS_INFO = [
    (
        "tracking_duration",
        "Tracking duration",
        "s",
    ),
    (
        "trajectory_curvature",
        "Trajectory curvature",
        "m^-1",
    ),
    (
        "perception_noise_savgol",
        "Perception noise",
        "m/s^2",
    ),
    (
        "motion_speed",
        "Motion speed",
        "m/s",
    ),
    (
        "minimum_distance_between_people",
        "Minimum human distance",
        "m",
    ),
]


def print_summary(
    title: str,
    summary: dict,
):
    """Print one metric summary."""

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    for key, label, unit in METRICS_INFO:

        result = summary[key]

        print(
            f"{label:30s} "
            f"mean = {result['mean']:.4f}, "
            f"median = {result['median']:.4f}, "
            f"std = {result['std']:.4f}, "
            # f"n = {result['n']}"
        )


print_summary(
    "OVERALL DATASET",
    overall_summary,
)

print_summary(
    "LOW-DENSITY DATASET",
    low_density_summary,
)

print_summary(
    "HIGH-DENSITY DATASET",
    high_density_summary,
)

print("\n" + "=" * 90)
print("SUMMARY SAVED")
print("=" * 90)
print(
    f"{output_file.resolve()}"
)