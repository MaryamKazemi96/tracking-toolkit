"""
Recompute the dataset-level THÖR metric summary.

Session 4 / Scenario 1 is excluded because its data are corrupted.

The summary is reconstructed from the per-recording mean, std,
and n values using pooled population statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INPUT = Path("results/thor_metrics")
OUTPUT = INPUT / "dataset_metrics.json"


# ---------------------------------------------------------------------
# Excluded recordings
# ---------------------------------------------------------------------

EXCLUDED_RECORDINGS = {
    (4, 1),
}


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

METRICS = [
    "tracking_duration",
    "trajectory_curvature",
    # "perception_noise",
    "motion_speed",
    "minimum_distance_between_people",
    "perception_noise_savgol",
# "motion_speed_nonoverlap",

# "motion_speed_frame_step",

]



def load_json(path: Path) -> dict:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def pooled_statistics(
    groups: list[tuple[float, float, int]],
) -> dict:
    """
    Pool mean/std/n from independent groups.

    Each group is represented as:

        (mean, std, n)

    The resulting std is the population standard deviation.
    """

    if not groups:

        return {
            "mean": None,
            "std": None,
            "n": 0,
        }

    total_n = sum(
        n
        for _, _, n in groups
    )

   
    mean = (
        sum(
            group_mean * n
            for group_mean, _, n in groups
        )
        / total_n
    )

 
    second_moment = (
        sum(
            n * (
                group_std ** 2
                + group_mean ** 2
            )
            for group_mean, group_std, n in groups
        )
        / total_n
    )


    variance = (
        second_moment
        - mean ** 2
    )

    variance = max(
        variance,
        0.0,
    )

    std = np.sqrt(
        variance
    )

    return {
        "mean": float(mean),
        "std": float(std),
        "n": int(total_n),
    }




def main():

    print("=" * 70)
    print("RECOMPUTING DATASET-LEVEL THÖR METRICS")
    print("=" * 70)

    groups = {
        metric: []
        for metric in METRICS
    }

    included = 0
    excluded = 0

    
    for session_dir in sorted(
        INPUT.glob("session_*")
    ):

        if not session_dir.is_dir():
            continue

        session = int(
            session_dir.name.split("_")[1]
        )

        for path in sorted(
            session_dir.glob("scenario_*.json")
        ):

            scenario = int(
                path.stem.split("_")[1]
            )

            
            if (
                session,
                scenario,
            ) in EXCLUDED_RECORDINGS:

                print(
                    f"EXCLUDED: "
                    f"Session {session}, "
                    f"Scenario {scenario}"
                )

                excluded += 1

                continue


            # Load recording metrics
            # -----------------------------------------------------

            data = load_json(path)

            included += 1

            print(
                f"Included: "
                f"Session {session}, "
                f"Scenario {scenario}"
            )


            for metric in METRICS:

                result = data.get(metric)

                if result is None:
                    continue

                mean = result.get("mean")
                std = result.get("std")
                n = result.get("n")

                if (
                    mean is None
                    or std is None
                    or n is None
                    or n == 0
                ):
                    continue

                groups[metric].append(
                    (
                        float(mean),
                        float(std),
                        int(n),
                    )
                )


    summary = {}

    for metric in METRICS:

        summary[metric] = pooled_statistics(
            groups[metric]
        )


    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )


    print("\n" + "=" * 70)
    print("CORRECTED DATASET SUMMARY")
    print("=" * 70)

    for metric in METRICS:

        result = summary[metric]

        print(
            f"{metric:40s} "
            f"{result['mean']:.4f} ± "
            f"{result['std']:.4f} "
            f"(n={result['n']})"
        )

    print("=" * 70)

    print(
        f"\nIncluded recordings: {included}"
    )

    print(
        f"Excluded recordings: {excluded}"
    )

    print(
        f"\nSaved corrected summary to:\n"
        f"{OUTPUT.resolve()}"
    )


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()