"""
Loading and validation for recordings.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when recordings.yaml is missing required fields or is inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load recordings.yaml from disk and validate it.

    Returns the parsed config as a plain dict. Raises ConfigError with a
    clear message if anything required is missing or malformed.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r") as f:
        config = yaml.safe_load(f)

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
   
    if not isinstance(config, dict):
        raise ConfigError("Top-level recordings.yaml content must be a mapping.")

    for key in ("dataset", "preprocessing", "sessions"):
        if key not in config:
            raise ConfigError(f"Missing required top-level key: '{key}'")

    dataset = config["dataset"]
    for key in ("fps", "up_axis", "coordinate_system"):
        if key not in dataset:
            raise ConfigError(f"Missing required key 'dataset.{key}'")

    if dataset["up_axis"] not in ("y", "z"):
        raise ConfigError(
            f"dataset.up_axis must be 'y' or 'z', got {dataset['up_axis']!r}"
        )

    if not isinstance(dataset["fps"], (int, float)) or dataset["fps"] <= 0:
        raise ConfigError(f"dataset.fps must be a positive number, got {dataset['fps']!r}")

    preprocessing = config["preprocessing"]
    if "waiting_area_x" not in preprocessing:
        raise ConfigError("Missing required key 'preprocessing.waiting_area_x'")

    sessions = config["sessions"]
    if not sessions:
        raise ConfigError("'sessions' must contain at least one session.")

    for session_id, session in sessions.items():
        if "scenarios" not in session or not session["scenarios"]:
            raise ConfigError(f"Session '{session_id}' has no scenarios defined.")

        for scenario_id, scenario in session["scenarios"].items():
            loc = f"sessions.{session_id}.scenarios.{scenario_id}"

            if "robot_present" not in scenario:
                raise ConfigError(f"{loc} is missing 'robot_present'")
            if not isinstance(scenario["robot_present"], bool):
                raise ConfigError(f"{loc}.robot_present must be true/false")

            if "humans" not in scenario or not scenario["humans"]:
                raise ConfigError(f"{loc} must have a non-empty 'humans' list")

            if scenario["robot_present"] and not scenario.get("robots"):
                raise ConfigError(
                    f"{loc} has robot_present=true but no 'robots' listed"
                )


def get_scenario_ids(config: dict[str, Any], session_id: str) -> list[int]:
    scenarios = config["sessions"][session_id]["scenarios"]
    return sorted(scenarios.keys())


def get_session_ids(config: dict[str, Any]) -> list[str]:
    return list(config["sessions"].keys())