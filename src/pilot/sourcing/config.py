"""Job board configuration loader from config/boards.yaml."""

from pathlib import Path
from typing import Any

import yaml


def get_default_boards_config_path() -> Path:
    """Return default path to config/boards.yaml."""
    current = Path(__file__).resolve()
    # Travel up until project root
    for parent in current.parents:
        candidate = parent / "config" / "boards.yaml"
        if candidate.is_file():
            return candidate
    return Path("config/boards.yaml")


def load_boards_config(config_path: Path | str | None = None) -> dict[str, list[dict[str, Any]]]:
    """
    Load configured job boards.

    Returns dict mapping source name ('greenhouse', 'ashby', 'lever') to list of
    board specs: [{'token': str, 'name': str}, ...]
    """
    path = Path(config_path) if config_path else get_default_boards_config_path()
    if not path.is_file():
        return {}

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    normalized: dict[str, list[dict[str, Any]]] = {}
    for source, entries in data.items():
        if not isinstance(entries, list):
            continue
        items: list[dict[str, Any]] = []
        for entry in entries:
            if isinstance(entry, str):
                items.append({"token": entry, "name": entry.capitalize()})
            elif isinstance(entry, dict):
                token = entry.get("token") or entry.get("org") or entry.get("slug")
                name = entry.get("name") or str(token).capitalize()
                if token:
                    items.append({"token": str(token), "name": str(name)})
        normalized[source.lower()] = items

    return normalized
