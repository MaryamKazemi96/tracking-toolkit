"""
Quality inspection utilities.
"""

from __future__ import annotations

import numpy as np





def detect_position_jumps(
    df,
    bodies,
    threshold=0.15,
):
    """
    Detect sudden position jumps.

    """

    results = {}

    for body in bodies:

        cols = [
            f"{body}.position.x",
            f"{body}.position.y",
            f"{body}.position.z",
        ]

        xyz = df[cols].to_numpy(float)
        frames = df["Frame"].to_numpy(int)
        time = df["Time"].to_numpy(float)

        valid = ~np.isnan(xyz).any(axis=1)

        xyz = xyz[valid]
        frames = frames[valid]
        time = time[valid]

        if len(xyz) < 2:
            results[body] = []
            continue

        distance = np.linalg.norm(
            np.diff(xyz, axis=0),
            axis=1,
        )

        jumps = []

        for i, d in enumerate(distance):

            if d < threshold:
                continue

            jumps.append(
                {
                    "frame": int(frames[i + 1]),
                    "time": float(time[i + 1]),
                    "distance": float(d),
                    "from": xyz[i],
                    "to": xyz[i + 1],
                }
            )

        results[body] = jumps

    return results




def detect_speed_outliers(
    df,
    bodies,
    threshold=3.0,
):
    """
    Detect unrealistically high speeds.

    """

    results = {}

    for body in bodies:

        cols = [
            f"{body}.position.x",
            f"{body}.position.y",
            f"{body}.position.z",
        ]

        xyz = df[cols].to_numpy(float)

        frames = df["Frame"].to_numpy(int)
        time = df["Time"].to_numpy(float)

        valid = ~np.isnan(xyz).any(axis=1)

        xyz = xyz[valid]
        frames = frames[valid]
        time = time[valid]

        if len(xyz) < 2:
            results[body] = []
            continue

        distance = np.linalg.norm(
            np.diff(xyz, axis=0),
            axis=1,
        )

        dt = np.diff(time)

        speed = distance / dt

        outliers = []

        for i, s in enumerate(speed):

            if s <= threshold:
                continue

            outliers.append(
                {
                    "frame": int(frames[i + 1]),
                    "time": float(time[i + 1]),
                    "speed": float(s),
                    "distance": float(distance[i]),
                }
            )

        results[body] = outliers

    return results




def print_quality_report(results):
    """
    Pretty-print inspection results.
    """

    for body, issues in results.items():

        print(f"\n{body}")

        if not issues:
            print("  ✓ No issues")
            continue

        print(f"  {len(issues)} issue(s)")

        for issue in issues:

            if "distance" in issue:

                print(
                    f"    Frame {issue['frame']:6d} | "
                    f"{issue['distance']:.3f} m jump"
                )

            elif "speed" in issue:

                print(
                    f"    Frame {issue['frame']:6d} | "
                    f"{issue['speed']:.2f} m/s"
                )

from pathlib import Path


def save_quality_report(
    results,
    output_file,
    title="Quality Report",
):
    """
    Save quality inspection results to a text file.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:

        f.write("=" * 80 + "\n")
        f.write(title + "\n")
        f.write("=" * 80 + "\n\n")

        total_issues = 0

        for body, issues in results.items():

            f.write(f"{body}\n")
            f.write("-" * len(body) + "\n")

            if not issues:
                f.write("No issues found.\n\n")
                continue

            f.write(f"{len(issues)} issue(s)\n\n")

            total_issues += len(issues)

            for issue in issues:

                if "distance" in issue:

                    f.write(
                        f"Frame {issue['frame']:6d} | "
                        f"Time {issue['time']:8.3f} s | "
                        f"Distance {issue['distance']:.3f} m | "
                    )

                elif "speed" in issue:

                    f.write(
                        f"Frame {issue['frame']:6d} | "
                        f"Time {issue['time']:8.3f} s | "
                        f"Speed {issue['speed']:.3f} m/s\n"
                    )

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write(f"Total issues: {total_issues}\n")