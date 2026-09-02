"""
Visualization utilities.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt





def _iter_agents(df, info):
    

    for body in info["humans"]:

        xyz = df[
            [
                f"{body}.position.x",
                f"{body}.position.y",
                f"{body}.position.z",
            ]
        ].to_numpy(float)

        time = df["Time"].to_numpy(float)

        valid = ~np.isnan(xyz).any(axis=1)

        yield (
            body,
            xyz[valid],
            time[valid],
            False,
        )

    for robot in info.get("robots", []):

        xyz = df[
            [
                f"{robot}.position.x",
                f"{robot}.position.y",
                f"{robot}.position.z",
            ]
        ].to_numpy(float)

        time = df["Time"].to_numpy(float)

        valid = ~np.isnan(xyz).any(axis=1)

        yield (
            robot,
            xyz[valid],
            time[valid],
            True,
        )





def plot_scene(df, info, ax=None):
    """
    Top-down trajectory plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    colors = plt.cm.tab10.colors
    color_id = 0

    for name, xyz, _, is_robot in _iter_agents(df, info):

        if len(xyz) == 0:
            continue

        if is_robot:

            ax.plot(
                xyz[:, 0],
                xyz[:, 1],
                "--",
                color="black",
                lw=2.5,
                label=name,
            )

        else:

            color = colors[color_id % len(colors)]
            color_id += 1

            ax.plot(
                xyz[:, 0],
                xyz[:, 1],
                color=color,
                lw=2,
                label=name,
            )

            ax.scatter(
                xyz[0, 0],
                xyz[0, 1],
                color=color,
                marker="o",
                edgecolor="black",
                s=45,
            )

            ax.scatter(
                xyz[-1, 0],
                xyz[-1, 1],
                color=color,
                marker="s",
                edgecolor="black",
                s=45,
            )

    ax.set_aspect("equal")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    ax.set_title(
        f"Session {info['session']}  Scenario {info['scenario']}"
    )

    ax.grid(True, alpha=0.3)

    ax.legend(fontsize=8)

    return fig, ax





def plot_x_time(df, info, ax=None):
    """
    X position versus time.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    colors = plt.cm.tab10.colors
    color_id = 0

    for name, xyz, time, is_robot in _iter_agents(df, info):

        if len(xyz) == 0:
            continue

        if is_robot:

            ax.plot(
                time,
                xyz[:, 0],
                "--",
                color="black",
                lw=2,
                label=name,
            )

        else:

            color = colors[color_id % len(colors)]
            color_id += 1

            ax.plot(
                time,
                xyz[:, 0],
                color=color,
                label=name,
            )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("X [m]")

    ax.grid(True, alpha=0.3)

    ax.legend(fontsize=8)

    return fig, ax




def plot_speed(df, info, fps, ax=None):
    """
    Plot speed computed over 1-second intervals.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    window = int(fps)  # 1 second

    colors = plt.cm.tab10.colors
    color_id = 0

    for name, xyz, time, is_robot in _iter_agents(df, info):

        if len(xyz) <= window:
            continue

        distance = np.linalg.norm(
            xyz[window:] - xyz[:-window],
            axis=1,
        )

        dt = time[window:] - time[:-window]

        speed = distance / dt

        plot_time = time[window:]

        if is_robot:

            ax.plot(
                plot_time,
                speed,
                "--",
                color="black",
                lw=2,
                label=name,
            )

        else:

            color = colors[color_id % len(colors)]
            color_id += 1

            ax.plot(
                plot_time,
                speed,
                color=color,
                lw=1.8,
                label=name,
            )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_ylim(bottom=0)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    return fig, ax
def plot_speedold(df, info, fps, ax=None):
    """
    Speed profiles.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    dt = 1.0 / fps

    colors = plt.cm.tab10.colors
    color_id = 0

    for name, xyz, time, is_robot in _iter_agents(df, info):

        if len(xyz) < 2:
            continue

        vel = np.gradient(
            xyz,
            dt,
            axis=0,
        )

        speed = np.linalg.norm(
            vel,
            axis=1,
        )

        if is_robot:

            ax.plot(
                time,
                speed,
                "--",
                color="black",
                lw=2,
                label=name,
            )

        else:

            color = colors[color_id % len(colors)]
            color_id += 1

            ax.plot(
                time,
                speed,
                color=color,
                label=name,
            )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")

    ax.grid(True, alpha=0.3)

    ax.legend(fontsize=8)

    return fig, ax





def plot_raw_vs_solved(raw_df, solved_df, info, ax=None):
    """
    Compare raw and solved trajectories.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    colors = plt.cm.tab10.colors

    for i, body in enumerate(info["humans"]):

        raw_xyz = raw_df[
            [
                f"{body}.position.x",
                f"{body}.position.y",
                f"{body}.position.z",
            ]
        ].to_numpy(float)

        solved_xyz = solved_df[
            [
                f"{body}.position.x",
                f"{body}.position.y",
                f"{body}.position.z",
            ]
        ].to_numpy(float)

        raw_valid = ~np.isnan(raw_xyz).any(axis=1)
        solved_valid = ~np.isnan(solved_xyz).any(axis=1)

        color = colors[i % len(colors)]

        ax.plot(
            raw_xyz[raw_valid, 0],
            raw_xyz[raw_valid, 1],
            "--",
            color="0.7",
            lw=1.5,
        )

        ax.plot(
            solved_xyz[solved_valid, 0],
            solved_xyz[solved_valid, 1],
            color=color,
            lw=2,
            label=body,
        )

    ax.set_aspect("equal")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    ax.set_title("Raw (gray dashed) vs Solved")

    ax.grid(True, alpha=0.3)

    ax.legend(fontsize=8)

    return fig, ax