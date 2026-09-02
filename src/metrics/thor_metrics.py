"""
metrics:

1. Tracking duration [s]
2. Trajectory curvature [m^-1]
3. Perception noise [m s^-2]
4. Motion speed [m s^-1]
5. Minimum distance between people [m]

"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter





def _position_columns(body: str) -> list[str]:
    """Return the 2D position columns used for the metrics."""

    return [
        f"{body}.position.x",
        f"{body}.position.y",
    ]





def _get_trajectory(
    df: pd.DataFrame,
    body: str,
) -> tuple[np.ndarray, np.ndarray]:
   

    cols = _position_columns(body)

    data = df[["Time"] + cols].to_numpy(float)

    valid = ~np.isnan(data).any(axis=1)

    data = data[valid]

    if len(data) == 0:
        return (
            np.empty(0),
            np.empty((0, 2)),
        )

    time = data[:, 0]
    position = data[:, 1:3]

    return time, position





def tracking_duration(
    df: pd.DataFrame,
    body: str,
) -> float | None:


    time, _ = _get_trajectory(df, body)

    if len(time) < 2:
        return None

    return float(time[-1] - time[0])




def _curvature_from_three_points(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float | None:


    d21 = np.linalg.norm(p2 - p1)
    d31 = np.linalg.norm(p3 - p1)
    d32 = np.linalg.norm(p3 - p2)

    denominator = d21 * d31 * d32

    if denominator <= 1e-12:
        return None

    numerator = abs(
        2.0
        * (
            (p2[0] - p1[0]) * (p3[1] - p1[1])
            - (p3[0] - p1[0]) * (p2[1] - p1[1])
        )
    )

    return float(numerator / denominator)





def trajectory_curvature(
    df: pd.DataFrame,
    body: str,
    segment_duration: float = 4.0,
    stride: float = 1.0,
) -> list[float]:


    time, position = _get_trajectory(df, body)

    if len(time) < 3:
        return []

    half_duration = segment_duration / 2.0

    start_time = time[0]
    end_time = time[-1]

    values = []

    current = start_time

    while current + segment_duration <= end_time:

        t1 = current
        t2 = current + half_duration
        t3 = current + segment_duration

        i1 = np.argmin(np.abs(time - t1))
        i2 = np.argmin(np.abs(time - t2))
        i3 = np.argmin(np.abs(time - t3))

        p1 = position[i1]
        p2 = position[i2]
        p3 = position[i3]

        curvature = _curvature_from_three_points(
            p1,
            p2,
            p3,
        )

        if curvature is not None:
            values.append(curvature)

        current += stride

    return values



def perception_noise(
    df: pd.DataFrame,
    body: str,
    fps: float,
    smoothing_window_seconds: float = 1.0,
    polynomial_order: int = 3,
) -> float | None:

    time, position = _get_trajectory(df, body)

    if len(time) < 5:
        return None

    window = int(round(
        smoothing_window_seconds * fps
    ))

    if window % 2 == 0:
        window += 1

    if window > len(position):
        window = len(position)

        if window % 2 == 0:
            window -= 1

    if window <= polynomial_order:
        return None

    smoothed_x = savgol_filter(
        position[:, 0],
        window_length=window,
        polyorder=polynomial_order,
    )

    smoothed_y = savgol_filter(
        position[:, 1],
        window_length=window,
        polyorder=polynomial_order,
    )

    dt = np.median(np.diff(time))

    if dt <= 0:
        return None

    acceleration_x = np.gradient(
        np.gradient(smoothed_x, dt),
        dt,
    )

    acceleration_y = np.gradient(
        np.gradient(smoothed_y, dt),
        dt,
    )

    acceleration = np.sqrt(
        acceleration_x**2
        + acceleration_y**2
    )

    return float(np.mean(np.abs(acceleration)))






def perception_noise_savgol(
    df: pd.DataFrame,
    body: str,
    fps: float,
    smoothing_window_seconds: float = 1.0,
    polynomial_order: int = 3,
) -> float | None:
  
    time, position = _get_trajectory(df, body)

    if len(time) < 5:
        return None

    window = int(round(
        smoothing_window_seconds * fps
    ))

    if window % 2 == 0:
        window += 1

    if window > len(position):
        window = len(position)

        if window % 2 == 0:
            window -= 1

    if window <= polynomial_order:
        return None

    dt = np.median(np.diff(time))

    if dt <= 0:
        return None

    acceleration_x = savgol_filter(
        position[:, 0],
        window_length=window,
        polyorder=polynomial_order,
        deriv=2,
        delta=dt,
    )

    acceleration_y = savgol_filter(
        position[:, 1],
        window_length=window,
        polyorder=polynomial_order,
        deriv=2,
        delta=dt,
    )

    acceleration = np.sqrt(
        acceleration_x**2
        + acceleration_y**2
    )

    return float(np.mean(acceleration))


def motion_speed(
    df: pd.DataFrame,
    body: str,
    interval: float = 1.0,
) -> list[float]:
   

    time, position = _get_trajectory(df, body)

    if len(time) < 2:
        return []

    values = []

    start = time[0]
    end = time[-1]

    while start + interval <= end:

        target = start + interval

        i1 = np.argmin(
            np.abs(time - start)
        )

        i2 = np.argmin(
            np.abs(time - target)
        )

        actual_dt = time[i2] - time[i1]

        if actual_dt <= 0:
            start += interval
            continue

        distance = np.linalg.norm(
            position[i2] - position[i1]
        )

        speed = distance / actual_dt

        values.append(float(speed))

        start += interval

    return values



def motion_speed_nonoverlap(
    df: pd.DataFrame,
    body: str,
    fps: float,
) -> list[float]:

    time, position = _get_trajectory(df, body)

    step = int(round(fps))

    if len(position) <= step:
        return []

    values = []

    for i in range(
        0,
        len(position) - step,
        step,
    ):

        j = i + step

        dt = time[j] - time[i]

        if dt <= 0:
            continue

        distance = np.linalg.norm(
            position[j] - position[i]
        )

        values.append(
            float(distance / dt)
        )

    return values



def motion_speed_frame_step(
    df: pd.DataFrame,
    body: str,
    fps: float,
) -> list[float]:
    """
    Alternative motion-speed implementation.

    Compute a 1-second displacement at EVERY frame.

    For 120 Hz data:

        frame 0   -> frame 120
        frame 1   -> frame 121
        frame 2   -> frame 122
        ...

    This corresponds directly to:

        disp = xyz[step:] - xyz[:-step]

    from the proposed implementation.
    """

    time, position = _get_trajectory(df, body)

    step = int(round(fps))

    if len(position) <= step:
        return []

    disp = position[step:] - position[:-step]

    dt = time[step:] - time[:-step]

    valid = dt > 0

    speed = (
        np.linalg.norm(disp[valid], axis=1)
        / dt[valid]
    )

    return speed.astype(float).tolist()



def minimum_distance_between_people(
    df: pd.DataFrame,
    humans: list[str],
) -> list[float]:
    """
    Compute the minimum pairwise distance between humans.

    At each frame, all humans with valid positions are considered.
    The smallest pairwise Euclidean distance is recorded.

    Robots are not included.
    """

    if len(humans) < 2:
        return []

    positions = {}

    for body in humans:

        cols = _position_columns(body)

        positions[body] = df[cols].to_numpy(float)

    values = []

    for frame_idx in range(len(df)):

        frame_positions = []

        for body in humans:

            p = positions[body][frame_idx]

            if np.isnan(p).any():
                continue

            frame_positions.append(p)

        if len(frame_positions) < 2:
            continue

        min_distance = np.inf

        for p1, p2 in combinations(
            frame_positions,
            2,
        ):

            distance = np.linalg.norm(
                p1 - p2
            )

            if distance < min_distance:
                min_distance = distance

        if np.isfinite(min_distance):
            values.append(float(min_distance))

    return values



def compute_recording_metrics(
    df: pd.DataFrame,
    humans: list[str],
    fps: float,
    curvature_stride: float = 1.0,
    smoothing_window_seconds: float = 1.0,
    smoothing_polynomial_order: int = 3,
) -> dict:
   

    tracking_values = []
    curvature_values = []

    noise_values = []
    noise_savgol_values = []

    speed_values = []
    speed_nonoverlap_values = []
    speed_frame_step_values = []

    for body in humans:

        duration = tracking_duration(
            df,
            body,
        )

        if duration is not None:
            tracking_values.append(duration)

        curvature_values.extend(
            trajectory_curvature(
                df,
                body,
                segment_duration=4.0,
                stride=curvature_stride,
            )
        )

        noise = perception_noise(
            df,
            body,
            fps=fps,
            smoothing_window_seconds=(
                smoothing_window_seconds
            ),
            polynomial_order=(
                smoothing_polynomial_order
            ),
        )

        if noise is not None:
            noise_values.append(noise)

        noise_savgol = perception_noise_savgol(
            df,
            body,
            fps=fps,
            smoothing_window_seconds=(
                smoothing_window_seconds
            ),
            polynomial_order=(
                smoothing_polynomial_order
            ),
        )

        if noise_savgol is not None:
            noise_savgol_values.append(
                noise_savgol
            )

        speed_values.extend(
            motion_speed(
                df,
                body,
                interval=1.0,
            )
        )
        speed_nonoverlap_values.extend(
            motion_speed_nonoverlap(
                df,
                body,
                fps=fps,
            )
        )
        speed_frame_step_values.extend(
            motion_speed_frame_step(
                df,
                body,
                fps=fps,
            )
        )
    distance_values = (
        minimum_distance_between_people(
            df,
            humans,
        )
    )

    return {
        "tracking_duration": tracking_values,
        "trajectory_curvature": curvature_values,
        "perception_noise": noise_values,
        "motion_speed": speed_values,
        "minimum_distance_between_people": distance_values,

        "perception_noise_savgol": noise_savgol_values,
        "motion_speed_nonoverlap": speed_nonoverlap_values,
        "motion_speed_frame_step": speed_frame_step_values,
    }


def summarize_metric(
    values: list[float],
) -> dict:
    """
    Calculate mean, std, median, and count for a metric.
    """

    if not values:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "n": 0,
        }

    array = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "n": int(len(array)),
    }




def summarize_dataset_metrics(
    raw_metrics: dict,
) -> dict:
    """
    Summarize pooled metric observations.

    Includes both original and alternative implementations.
    """

    result = {}

    result["tracking_duration"] = summarize_metric(
        raw_metrics["tracking_duration"]
    )

    result["trajectory_curvature"] = summarize_metric(
        raw_metrics["trajectory_curvature"]
    )

    result["perception_noise"] = summarize_metric(
        raw_metrics["perception_noise_savgol"]
    )

    result["motion_speed"] = summarize_metric(
        raw_metrics["motion_speed"]
    )

    result["minimum_distance_between_people"] = (
        summarize_metric(
            raw_metrics[
                "minimum_distance_between_people"
            ]
        )
    )

    result["perception_noise_savgol"] = summarize_metric(
        raw_metrics["perception_noise_savgol"]
    )


    # result["motion_speed_nonoverlap"] = summarize_metric(
    #     raw_metrics["motion_speed_nonoverlap"]
    # )

    # result["motion_speed_frame_step"] = summarize_metric(
    #     raw_metrics["motion_speed_frame_step"]
    # )

    return result