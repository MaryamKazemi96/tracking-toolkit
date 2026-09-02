"""
Trajectory trimming utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------


def trim(
    df: pd.DataFrame,
    presence: dict[str, dict[str, int | None]],
) -> pd.DataFrame:
    """
    Trim trajectories using detected presence intervals.

    """

    df = df.copy()

    frames = df["Frame"]

    for body, interval in presence.items():

        start = interval["start"]
        end = interval["end"]

        if start is None or end is None:
            continue

        body_columns = [
            c for c in df.columns
            if c.startswith(body + ".")
        ]

        if not body_columns:
            continue

        mask = (frames < start) | (frames > end)

        df.loc[mask, body_columns] = np.nan

    return df