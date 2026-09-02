"""
OptiTrack CSV parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd





@dataclass
class RecordingMetadata:
    take_name: str
    capture_frame_rate: float
    export_frame_rate: float
    total_frames: int
    rotation_type: str
    length_units: str
    coordinate_space: str





class OptiTrackParser:
    """
    Parse one OptiTrack CSV file.
    """

    HEADER_START = 2
    HEADER_ROWS = 5
    DATA_START = 7

    def __init__(self, filepath):

        self.filepath = Path(filepath)

    

    def parse(self):

        metadata = self._parse_metadata()

        df = pd.read_csv(
            self.filepath,
            skiprows=self.DATA_START,
            header=None,
            low_memory=False,
        )

        df.columns = self._build_columns()

        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        return metadata, df

    

    def _parse_metadata(self):

        with open(self.filepath, encoding="utf-8") as f:
            first = f.readline().strip().split(",")

        meta = {}

        for i in range(0, len(first) - 1, 2):
            meta[first[i]] = first[i + 1]

        return RecordingMetadata(
            take_name=meta["Take Name"],
            capture_frame_rate=float(meta["Capture Frame Rate"]),
            export_frame_rate=float(meta["Export Frame Rate"]),
            total_frames=int(meta["Total Frames in Take"]),
            rotation_type=meta["Rotation Type"],
            length_units=meta["Length Units"],
            coordinate_space=meta["Coordinate Space"],
        )

    

    def _build_columns(self):

        header = pd.read_csv(
            self.filepath,
            skiprows=self.HEADER_START,
            nrows=self.HEADER_ROWS,
            header=None,
            low_memory=False,
        )

        columns = []

        for col in range(header.shape[1]):

            if col == 0:
                columns.append("Frame")
                continue

            if col == 1:
                columns.append("Time")
                continue

            object_name = str(header.iloc[1, col]).strip()
            data_type = str(header.iloc[3, col]).strip()
            axis = header.iloc[4, col]

            object_name = (
                object_name
                .replace(" ", "")
                .replace(":", ".")
            )

            data_type = (
                data_type
                .replace(" ", "_")
                .lower()
            )

            if pd.isna(axis):

                columns.append(
                    f"{object_name}.{data_type}"
                )

            else:

                columns.append(
                    f"{object_name}.{data_type}.{str(axis).lower()}"
                )

        return columns