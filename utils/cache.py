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
import os
import pickle
import threading
from pathlib import Path
from typing import Any

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = Path("./.quant_cache")

# Bump whenever the data layer changes the meaning or layout of what it
# produces (columns, adjustment convention, PIT handling, ...). Cached stages
# written under an older schema must not be replayed as if identical.
# 4: benchmarks aligned to the forward-return convention; metadata may carry
#    a PIT sector panel.
# 5: synthetic panels use the exchange calendar (exchange holidays removed
#    from prices/benchmark/financial visibility), suspension flags are derived
#    from the data, and the tradability mask covers listing/delisting/
#    suspension. Panels cached under 4 are business-day grids and must not be
#    replayed.
DATA_SCHEMA_VERSION = 5

# Bump when the key recipe itself changes, so entries written by an older
# recipe cannot collide with new ones.
CACHE_KEY_VERSION = 2


class PipelineCache:
    """Caches pipeline stages to disk for fast re-runs.

    Cache key = MD5 hash of the config subset plus everything that changes the
    produced data but lives outside the YAML: the data provider actually used
    (a silent fallback to synthetic must not share the requested provider's
    slot), explicit CLI/environment overrides, and the data schema version.
    """

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_path(self, stage: str, cache_key: str) -> Path:
        return self.cache_dir / f"{stage}_{cache_key}.pkl"

    @staticmethod
    def make_config_hash(
        config_dict: dict[str, Any],
        provider: str | None = None,
        overrides: dict[str, Any] | None = None,
        schema_version: int | str | None = None,
    ) -> str:
        """Create a deterministic hash from everything that determines a run.

        Args:
            config_dict: The raw config mapping (as loaded from YAML).
            provider: The provider that will actually produce the data. Pass
                the *effective* provider -- if a silent fallback to synthetic
                happened, hashing the requested provider would make the run
                read the requested provider's cached output.
            overrides: Explicit overrides applied on top of the YAML (CLI
                flags, environment-derived settings). Their presence and values
                both matter, so ``{"use_baostock": False}`` and no overrides at
                all produce different keys only if they change behaviour.
            schema_version: Data-layer schema version; defaults to
                ``DATA_SCHEMA_VERSION``.

        The raw YAML alone is not a run identity: ``--use-baostock``, an
        environment token enabling a real provider, a silent fallback, or a
        code change to the data schema all alter the output while leaving the
        YAML untouched.
        """
        payload = {
            "key_version": CACHE_KEY_VERSION,
            "config": config_dict,
            "provider": provider,
            "overrides": overrides or {},
            "schema_version": (
                DATA_SCHEMA_VERSION if schema_version is None else schema_version
            ),
        }
        config_str = json.dumps(payload, sort_keys=True, default=str)
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
                # Another writer may hold the file on Windows; a failed
                # cleanup must not escape as PermissionError.
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        return None

    def save_stage(self, stage: str, cache_key: str, data: Any) -> None:
        """Save a stage to cache atomically.

        Writing the target in place let a concurrent reader deserialize a
        half-written pickle (or hit a Windows sharing violation). Write to a
        unique temp file, fsync, then os.replace.
        """
        path = self._make_path(stage, cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(
            f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            with open(tmp, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            logger.info("Cached: %s (%s)", stage, cache_key)
        except Exception as e:
            logger.debug("Failed to save cache %s: %s", stage, e)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

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
