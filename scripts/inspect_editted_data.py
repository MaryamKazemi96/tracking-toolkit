"""
Per-rigid-body diagnostic metrics for a suspicious recording.

Used to identify corrupted trajectories and locate suspicious
timestamps within the trajectory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.io.loader import DatasetLoader
from src.preprocessing.presence import detect_presence
from src.preprocessing.trim import trim

from src.metrics.thor_metrics import (
    tracking_duration,
    trajectory_curvature,
    perception_noise,
    motion_speed,
)



SESSION = 1
SCENARIO = 3

OUTPUT = Path(
    f"results/corruption_check/"
    f"session_{SESSION}_scenario_{SCENARIO}"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

SPEED_THRESHOLD = 3.0
ACCELERATION_THRESHOLD = 20.0


loader = DatasetLoader(
    root="data/OptiTrack/edited_data/solved",
    config="config/recordings.yaml",
)
loader_raw = DatasetLoader(
    root="data/OptiTrack/edited_data/raw",
    config="config/recordings.yaml",
)

fps = loader.config["dataset"]["fps"]

waiting_area_x = (
    loader.config["preprocessing"]
    ["waiting_area_x"]
)


_, df, info = loader.load(
    session=SESSION,
    scenario=SCENARIO,
)
_, df_raw, info_raw = loader_raw.load(
    session=SESSION,
    scenario=SCENARIO,
)

humans = info["humans"]

robots = info["robots"]

bodies = humans + robots


presence = detect_presence(
    df=df_raw,
    bodies=bodies,
    waiting_area_x=waiting_area_x,
)
print(presence)
df = trim(
    df,
    presence,
)


print("\n" + "=" * 90)
print(
    f"SESSION {SESSION} / SCENARIO {SCENARIO}"
)
print("=" * 90)

print(f"FPS:       {fps}")
print(f"Frames:    {len(df)}")
print(f"Duration:  {df['Time'].iloc[-1]:.3f} s")

print("\nHumans:")
for body in humans:
    print(f"  {body}")

print("\nRobots:")
for body in robots:
    print(f"  {body}")


for body in humans:

    print("\n" + "=" * 90)
    print(body)
    print("=" * 90)

    cols = [
        f"{body}.position.x",
        f"{body}.position.y",
    ]

    data = df[
        ["Frame", "Time"] + cols
    ].to_numpy(float)

    valid = ~np.isnan(
        data[:, 2:]
    ).any(axis=1)

    data = data[valid]

    if len(data) < 3:

        print("Not enough valid trajectory data.")
        continue

    frames = data[:, 0].astype(int)
    time = data[:, 1]
    xy = data[:, 2:4]

    print("\nPOSITION")

    print(
        f"  X range: "
        f"{xy[:, 0].min():.4f} "
        f"to "
        f"{xy[:, 0].max():.4f} m"
    )

    print(
        f"  Y range: "
        f"{xy[:, 1].min():.4f} "
        f"to "
        f"{xy[:, 1].max():.4f} m"
    )


    duration = tracking_duration(
        df,
        body,
    )

    print("\nTRACKING")

    print(
        f"  Duration: "
        f"{duration:.3f} s"
    )

    print(
        f"  Samples:  "
        f"{len(xy)}"
    )


    dt = np.diff(time)

    displacement = np.linalg.norm(
        np.diff(xy, axis=0),
        axis=1,
    )

    speed = displacement / dt

    print("\nSPEED")

    print(
        f"  Mean:   "
        f"{np.mean(speed):.6f} m/s"
    )

    print(
        f"  Median: "
        f"{np.median(speed):.6f} m/s"
    )

    print(
        f"  Std:    "
        f"{np.std(speed):.6f} m/s"
    )

    print(
        f"  Max:    "
        f"{np.max(speed):.6e} m/s"
    )

    print(
        f"  P99:    "
        f"{np.percentile(speed, 99):.6f} m/s"
    )

    suspicious_speed = np.where(
        speed > SPEED_THRESHOLD
    )[0]

    print(
        f"  > {SPEED_THRESHOLD:.1f} m/s: "
        f"{len(suspicious_speed)} samples"
    )


    print("\nLARGEST SPEED SPIKES")

    indices = np.argsort(
        speed
    )[-10:][::-1]

    for i in indices:

        print(
            f"  frame={frames[i+1]:7d} | "
            f"time={time[i+1]:10.4f} s | "
            f"speed={speed[i]:12.4e} m/s | "
            f"distance={displacement[i]:12.4e} m | "
            f"dt={dt[i]:.6f} s"
        )

    acceleration = np.diff(speed) / np.diff(
        time[:-1]
    )

    abs_acceleration = np.abs(
        acceleration
    )

    print("\nACCELERATION")

    print(
        f"  Mean absolute: "
        f"{np.mean(abs_acceleration):.6f} m/s²"
    )

    print(
        f"  Median absolute: "
        f"{np.median(abs_acceleration):.6f} m/s²"
    )

    print(
        f"  Max absolute: "
        f"{np.max(abs_acceleration):.6e} m/s²"
    )

    suspicious_acceleration = np.where(
        abs_acceleration > ACCELERATION_THRESHOLD
    )[0]

    print(
        f"  > {ACCELERATION_THRESHOLD:.1f} m/s²: "
        f"{len(suspicious_acceleration)} samples"
    )


    print("\nLARGEST ACCELERATION SPIKES")

    indices = np.argsort(
        abs_acceleration
    )[-10:][::-1]

    for i in indices:

        print(
            f"  frame={frames[i+2]:7d} | "
            f"time={time[i+2]:10.4f} s | "
            f"acc={acceleration[i]:12.4e} m/s²"
        )

   
    curvature = trajectory_curvature(
        df,
        body,
        segment_duration=4.0,
        stride=1.0,
    )

    noise = perception_noise(
        df,
        body,
        fps=fps,
        smoothing_window_seconds=1.0,
        polynomial_order=3,
    )

    thor_speed = motion_speed(
        df,
        body,
        interval=1.0,
    )

    print("\nTHÖR METRICS")

    print(
        f"  Curvature:"
    )

    print(
        f"    mean   = "
        f"{np.mean(curvature):.6f}"
    )

    print(
        f"    median = "
        f"{np.median(curvature):.6f}"
    )

    print(
        f"    max    = "
        f"{np.max(curvature):.6e}"
    )

    print(
        f"  Perception noise:"
    )

    print(
        f"    {noise:.6e} m/s²"
    )

    print(
        f"  Motion speed:"
    )

    print(
        f"    mean   = "
        f"{np.mean(thor_speed):.6f} m/s"
    )

    print(
        f"    median = "
        f"{np.median(thor_speed):.6f} m/s"
    )

    print(
        f"    max    = "
        f"{np.max(thor_speed):.6e} m/s"
    )

   

    if len(suspicious_speed) > 0:

        rows = []

        for i in suspicious_speed:

            rows.append(
                {
                    "frame": frames[i + 1],
                    "time": time[i + 1],
                    "x": xy[i + 1, 0],
                    "y": xy[i + 1, 1],
                    "speed_m_s": speed[i],
                    "distance_m": displacement[i],
                    "dt_s": dt[i],
                }
            )

        suspicious_df = pd.DataFrame(
            rows
        )

        suspicious_df.to_csv(
            OUTPUT
            / f"{body}_speed_anomalies.csv",
            index=False,
        )



print("\n" + "=" * 90)
print("SUMMARY PER RIGID BODY")
print("=" * 90)

rows = []

for body in humans:

    cols = [
        f"{body}.position.x",
        f"{body}.position.y",
    ]

    data = df[
        ["Frame", "Time"] + cols
    ].to_numpy(float)

    valid = ~np.isnan(
        data[:, 2:]
    ).any(axis=1)

    data = data[valid]

    if len(data) < 3:
        continue

    time = data[:, 1]
    xy = data[:, 2:4]

    dt = np.diff(time)

    displacement = np.linalg.norm(
        np.diff(xy, axis=0),
        axis=1,
    )

    speed = displacement / dt

    curvature = trajectory_curvature(
        df,
        body,
        segment_duration=4.0,
        stride=1.0,
    )

    noise = perception_noise(
        df,
        body,
        fps=fps,
        smoothing_window_seconds=1.0,
        polynomial_order=3,
    )

    rows.append(
        {
            "body": body,
            "duration_s": time[-1] - time[0],
            "samples": len(xy),
            "speed_mean": np.mean(speed),
            "speed_median": np.median(speed),
            "speed_max": np.max(speed),
            "speed_gt_3mps": np.sum(
                speed > SPEED_THRESHOLD
            ),
            "acceleration_median": np.median(
                np.abs(
                    np.diff(speed)
                    / np.diff(time[:-1])
                )
            ),
            "curvature_mean": np.mean(
                curvature
            ),
            "curvature_median": np.median(
                curvature
            ),
            "perception_noise": noise,
            "x_min": xy[:, 0].min(),
            "x_max": xy[:, 0].max(),
            "y_min": xy[:, 1].min(),
            "y_max": xy[:, 1].max(),
        }
    )

summary = pd.DataFrame(rows)

summary.to_csv(
    OUTPUT / "per_body_summary.csv",
    index=False,
)

print(
    summary.to_string(
        index=False
    )
)

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)

print(
    f"Results saved to:\n{OUTPUT.resolve()}"
)