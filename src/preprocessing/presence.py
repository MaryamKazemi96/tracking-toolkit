"""
Presence detection utilities.

Detect when each rigid body is actually inside the tracking area.
"""

from __future__ import annotations

import numpy as np
import pandas as pd





def detect_presence(
    df: pd.DataFrame,
    bodies: list[str],
    waiting_area_x: float,
    min_duration: int = 30,
) -> dict[str, dict[str, int | None]]:


    frames = df["Frame"].to_numpy()

    presence = {}

    for body in bodies:
        # print(f"Detecting presence for body {body}...")
        cols = [
            f"{body}.position.x",
            f"{body}.position.y",
            f"{body}.position.z",
        ]
        # print(df.columns)
        xyz = df[cols].to_numpy(float)
        # print(xyz[:15,:])
        tracked = ~np.isnan(xyz).any(axis=1)

        inside = xyz[:, 0] > waiting_area_x
        # print(f"Body {body}: {np.sum(inside)} frames inside waiting area")
        present = tracked & inside

        intervals = _find_intervals(
            present,
            min_duration,
        )

        if not intervals:

            presence[body] = {
                "start": None,
                "end": None,
            }

            continue

        start_idx = intervals[0][0]
        end_idx = intervals[-1][1]

        presence[body] = {
            "start": int(frames[start_idx]),
            "end": int(frames[end_idx]),
        }

    return presence





def _find_intervals(
    mask: np.ndarray,
    min_duration: int,
) -> list[tuple[int, int]]:
   
    intervals = []

    start = None

    for i, value in enumerate(mask):

        if value and start is None:
            start = i

        elif not value and start is not None:

            end = i - 1

            if end - start + 1 >= min_duration:
                intervals.append((start, end))

            start = None

    if start is not None:

        end = len(mask) - 1

        if end - start + 1 >= min_duration:
            intervals.append((start, end))

    return intervals