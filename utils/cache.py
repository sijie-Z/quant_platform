"""Pipeline result caching for fast iteration.

Caches intermediate results (data pipeline output, factors, alpha signal)
to avoid recomputation when running the same config repeatedly. Uses pickle
with config hash as cache key.

Usage:
    from quant_platform.utils.cache import PipelineCache

    cache = PipelineCache("./.cache")
    data = cache.load_stage("data", config_hash)
    if data is None:
        data = run_data_pipeline()
        cache.save_stage("data", config_hash, data)
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = Path("./.quant_cache")


class PipelineCache:
    """Caches pipeline stages to disk for fast re-runs.

    Cache key = MD5 hash of the relevant config subset, ensuring
    that config changes invalidate the cache automatically.
    """

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_path(self, stage: str, cache_key: str) -> Path:
        return self.cache_dir / f"{stage}_{cache_key}.pkl"

    @staticmethod
    def make_config_hash(config_dict: dict[str, Any]) -> str:
        """Create a deterministic hash from config dict."""
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def load_stage(self, stage: str, cache_key: str) -> Any | None:
        """Load a cached stage. Returns None if not found."""
        path = self._make_path(stage, cache_key)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                logger.info("Cache hit: %s (%s)", stage, cache_key)
                return data
            except Exception as e:
                logger.debug("Failed to load cache %s: %s", stage, e)
                path.unlink(missing_ok=True)
        return None

    def save_stage(self, stage: str, cache_key: str, data: Any) -> None:
        """Save a stage to cache."""
        path = self._make_path(stage, cache_key)
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Cached: %s (%s)", stage, cache_key)
        except Exception as e:
            logger.debug("Failed to save cache %s: %s", stage, e)
            path.unlink(missing_ok=True)

    def clear(self) -> int:
        """Clear all cached files. Returns number of files removed."""
        count = 0
        for p in self.cache_dir.glob("*.pkl"):
            p.unlink()
            count += 1
        logger.info("Cleared %d cache files", count)
        return count

    def list_cached(self) -> list[str]:
        """List cached stages."""
        return sorted(p.stem for p in self.cache_dir.glob("*.pkl"))
