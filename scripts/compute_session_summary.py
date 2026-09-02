"""
Compute the recording-level summary table used in the paper
(Table III: per-session overview, Table IV: per-session-per-scenario detail).

For every (session, scenario) recording this script reports:

    - density              (low / high, from config)
    - n_participants        (number of humans configured for that scenario)
    - duration_s            (wall-clock duration the scenario was actually recorded)
    - human_tracks          (number of humans with a non-empty presence interval)
    - robot_tracks          (number of robots with a non-empty presence interval)
    - frames                (number of motion-capture frames spanned by the recording)

"duration_s" is computed from *presence*, not from the raw take length: it is the
time between the first body entering the tracking volume and the last body
leaving it (matching how a participant/robot is actually "in the scene").
"frames" is the corresponding frame span (end_frame - start_frame + 1) using the
Frame column, so it lines up with the "Frames (120 Hz)" column in Table III/IV.

Usage
-----
    python scripts/compute_session_summary.py

Outputs
-------
    results/session_summary/scenario_summary.csv   (one row per session x scenario -> Table IV)
    results/session_summary/session_summary.csv     (one row per session          -> Table III)
    results/session_summary/tables.tex              (LaTeX-ready rows for both tables)

Run this from the repository root (same place you run the other scripts, e.g.
scripts/compute_thor_metrics.py) so that "data/..." and "config/..." resolve.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.io.loader import DatasetLoader
from src.preprocessing.presence import detect_presence

# ---------------------------------------------------------------------
# Paths — adjust ROOT_RAW if your raw data lives somewhere else.
# Presence is always detected on the RAW data (matches check_quality.py /
# compute_thor_metrics.py convention), since "solved" data has already been
# gap-filled and no longer reflects when a body truly entered/left the room.
# ---------------------------------------------------------------------

ROOT_RAW = "data/OptiTrack/raw"
CONFIG = "config/recordings.yaml"

OUTPUT = Path("results/session_summary")
OUTPUT.mkdir(parents=True, exist_ok=True)


def session_scenario_stats(loader: DatasetLoader, session: int, scenario: int) -> dict:
    """Compute one row of Table IV for a single (session, scenario)."""

    _, df, info = loader.load(session=session, scenario=scenario)

    bodies = info["humans"] + info["robots"]

    presence = detect_presence(
        df=df,
        bodies=bodies,
        waiting_area_x=loader.config["preprocessing"]["waiting_area_x"],
    )

    fps = loader.config["dataset"]["fps"]

    starts = [v["start"] for v in presence.values() if v["start"] is not None]
    ends = [v["end"] for v in presence.values() if v["end"] is not None]

    if starts and ends:
        first_frame = min(starts)
        last_frame = max(ends)
        frames = last_frame - first_frame + 1
        duration_s = frames / fps
    else:
        first_frame = None
        last_frame = None
        frames = 0
        duration_s = 0.0

    human_tracks = sum(
        1 for h in info["humans"] if presence[h]["start"] is not None
    )
    robot_tracks = sum(
        1 for r in info["robots"] if presence[r]["start"] is not None
    )

    return {
        "session": session,
        "scenario": scenario,
        "density": loader.config["sessions"][f"session_{session}"]["density"],
        "n_configured_humans": len(info["humans"]),
        "human_tracks": human_tracks,
        "robot_tracks": robot_tracks,
        "duration_s": round(duration_s, 1),
        "frames": frames,
        "first_frame": first_frame,
        "last_frame": last_frame,
    }


def main() -> None:

    loader = DatasetLoader(root=ROOT_RAW, config=CONFIG)

    scenario_rows: list[dict] = []

    for session_name, session_cfg in loader.config["sessions"].items():

        session = int(session_name.split("_")[1])

        for scenario in sorted(session_cfg["scenarios"]):

            print(f"Session {session}  Scenario {scenario} ...", end=" ")

            row = session_scenario_stats(loader, session, scenario)

            scenario_rows.append(row)

            print(
                f"duration={row['duration_s']:.1f}s  "
                f"humans={row['human_tracks']}  "
                f"robots={row['robot_tracks']}  "
                f"frames={row['frames']}"
            )

    # --- write per-scenario CSV (Table IV) ---------------------------------

    scenario_csv = OUTPUT / "scenario_summary.csv"

    with open(scenario_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scenario_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scenario_rows)

    # --- aggregate to per-session CSV (Table III) ---------------------------
    # Session duration = sum of the 4 scenario durations (total recording time).
    # human_tracks / robot_tracks reported as the max observed across that
    # session's scenarios (a body only needs to appear once to count).

    sessions: dict[int, dict] = {}

    for row in scenario_rows:

        s = row["session"]

        if s not in sessions:
            sessions[s] = {
                "session": s,
                "density": row["density"],
                "n_participants": row["n_configured_humans"],
                "duration_s": 0.0,
                "human_tracks": 0,
                "robot_tracks": 0,
                "frames": 0,
            }

        sessions[s]["duration_s"] += row["duration_s"]
        sessions[s]["human_tracks"] = max(sessions[s]["human_tracks"], row["human_tracks"])
        sessions[s]["robot_tracks"] = max(sessions[s]["robot_tracks"], row["robot_tracks"])
        sessions[s]["frames"] += row["frames"]

    session_rows = [sessions[s] for s in sorted(sessions)]

    for row in session_rows:
        row["duration_s"] = round(row["duration_s"], 1)

    session_csv = OUTPUT / "session_summary.csv"

    with open(session_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(session_rows[0].keys()))
        writer.writeheader()
        writer.writerows(session_rows)

    # --- write LaTeX-ready rows so you can paste straight into Table III/IV -

    tex_path = OUTPUT / "tables.tex"

    with open(tex_path, "w", encoding="utf-8") as f:

        f.write("% ---- Table III rows (per-session) ----\n")
        for row in session_rows:
            f.write(
                f"{row['session']} & {row['density'].capitalize()} & "
                f"{row['n_participants']} & {row['duration_s']:.1f} & "
                f"{row['human_tracks']} & {row['robot_tracks']} & "
                f"{row['frames']} \\\\\n"
            )

        f.write("\n% ---- Table IV rows (per-session-per-scenario) ----\n")
        for row in scenario_rows:
            f.write(
                f"{row['session']} & {row['density'].capitalize()} & "
                f"{row['n_configured_humans']} & S{row['scenario']} & "
                f"{row['duration_s']:.1f} & {row['human_tracks']} & "
                f"{row['robot_tracks']} & {row['frames']} \\\\\n"
            )

    print("\nDone.")
    print(f"  Per-scenario (Table IV): {scenario_csv.resolve()}")
    print(f"  Per-session  (Table III): {session_csv.resolve()}")
    print(f"  LaTeX rows:               {tex_path.resolve()}")


if __name__ == "__main__":
    main()