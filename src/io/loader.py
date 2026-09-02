"""
Dataset loader.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .parser import OptiTrackParser


class DatasetLoader:
    """
    Load one recording from the dataset.
    """

    def __init__(
        self,
        root,
        config,
    ):

        self.root = Path(root)

        with open(config, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    

    def load(
        self,
        session,
        scenario,
    ):

        csv_file = self._find_csv(
            session,
            scenario,
        )

        metadata, df = OptiTrackParser(csv_file).parse()

        self._normalize_coordinates(df)

        info = self._load_info(
            session,
            scenario,
        )

        return metadata, df, info

    

    def _find_csv(
        self,
        session,
        scenario,
    ):

        folder = self.root / f"session {session}"

        files = list(
            folder.glob(f"*-{scenario}.csv")
        )

        if len(files) != 1:

            raise FileNotFoundError(
                f"Expected one CSV for "
                f"session={session}, "
                f"scenario={scenario}. "
                f"Found {len(files)}."
            )

        return files[0]

    

    def _load_info(
        self,
        session,
        scenario,
    ):

        cfg = (
            self.config["sessions"]
            [f"session_{session}"]
            ["scenarios"]
            [scenario]
        )

        humans = [
            h.replace(" ", "")
            for h in cfg["humans"]
        ]

        robots = [
            r.replace(" ", "")
            for r in cfg.get("robots", [])
        ]

        return {
            "session": session,
            "scenario": scenario,
            "robot_present": cfg["robot_present"],
            "humans": humans,
            "robots": robots,
        }

    

    def _normalize_coordinates(self, df):

        up = self.config["dataset"]["up_axis"].lower()

        if up == "z":
            return

        if up == "y":

            for col in df.columns:

                if col.endswith(".position.y"):

                    zcol = col[:-1] + "z"

                    df[col], df[zcol] = (
                        df[zcol].copy(),
                        df[col].copy(),
                    )

        elif up == "x":

            for col in df.columns:

                if col.endswith(".position.x"):

                    zcol = col[:-1] + "z"

                    df[col], df[zcol] = (
                        df[zcol].copy(),
                        df[col].copy(),
                    )

        else:

            raise ValueError(
                f"Unknown up_axis '{up}'."
            )