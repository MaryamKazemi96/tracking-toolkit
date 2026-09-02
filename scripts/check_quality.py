"""
Inspect the entire dataset for trajectory quality issues.
"""

from pathlib import Path

from src.io.loader import DatasetLoader

from src.preprocessing.presence import detect_presence
from src.preprocessing.trim import trim

from src.inspect.quality_check import (
    detect_position_jumps,
    detect_speed_outliers,
    save_quality_report,
)

# ---------------------------------------------------------------------

OUTPUT = Path("results/anomalies")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------

loader =  DatasetLoader(
    root="data/OptiTrack/solved",
    config="config/recordings.yaml",
)

for session_name, session_cfg in loader.config["sessions"].items():

    session = int(session_name.split("_")[1])

    for scenario in session_cfg["scenarios"]:

        print(f"Checking Session {session}  Scenario {scenario}")

        _, df, info = loader.load(
            session=session,
            scenario=scenario,
        )

        bodies = info["humans"] + info["robots"]

        presence = detect_presence(
            df=df,
            bodies=bodies,
            waiting_area_x=loader.config["preprocessing"]["waiting_area_x"],
        )

        df = trim(df, presence)

        position_jumps = detect_position_jumps(
            df,
            bodies,
        )

        save_quality_report(
            position_jumps,
            OUTPUT
            / f"session_{session}_scenario_{scenario}_position_jumps.txt",
        )

        speed_outliers = detect_speed_outliers(
            df,
            bodies,
        )

        save_quality_report(
            speed_outliers,
            OUTPUT
            / f"session_{session}_scenario_{scenario}_speed_outliers.txt",
        )

print("\nDone.")
print(f"Reports saved to: {OUTPUT.resolve()}")