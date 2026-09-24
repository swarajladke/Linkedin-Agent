"""Local disk cache for raw ingestion responses and parsed artifacts."""

import hashlib
import json
from pathlib import Path
from typing import Any


class IngestionCache:
    """Filesystem-based cache for raw external requests and parsed documents."""

    def __init__(self, cache_dir: str | Path = ".cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, namespace: str, key: str) -> Path:
        ns_dir = self.cache_dir / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return ns_dir / f"{key_hash}.json"

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        """Retrieve cached JSON payload if present."""
        path = self._get_path(namespace, key)
        if not path.is_file():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, namespace: str, key: str, payload: dict[str, Any]) -> None:
        """Store JSON payload in cache."""
        path = self._get_path(namespace, key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception:
            pass
