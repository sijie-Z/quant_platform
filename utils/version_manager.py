"""Configuration version manager.

Inspired by wudao-hero-skill's version_manager.py, this module adds
explicit versioning, rollback, and diff capabilities to configuration
management. Each version is a full snapshot of the active config.

Versions are stored as YAML files in a versions directory with a
JSON metadata index.

Usage:
    mgr = VersionManager()
    v = mgr.save(config_dict, description="Added growth factor")
    mgr.list()
    mgr.diff("v1", "v3")
    mgr.rollback("v2")
"""

from __future__ import annotations

import difflib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

_VERSION_FILENAME = "version.yaml"
_META_FILENAME = "index.json"
_DEFAULT_DIR = ".quant_versions"


@dataclass
class ConfigVersion:
    """Metadata for a single configuration version.

    Args:
        id: Version identifier, e.g. "v1", "v2".
        timestamp: When the version was created.
        description: Human-readable description of what changed.
        run_id: Run ID if this version was auto-saved during a pipeline run.
        config_hash: MD5 hash of the config for quick comparison.
    """
    id: str
    timestamp: str  # ISO format
    description: str = ""
    run_id: str = ""
    config_hash: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "description": self.description,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> ConfigVersion:
        return cls(
            id=d["id"],
            timestamp=d["timestamp"],
            description=d.get("description", ""),
            run_id=d.get("run_id", ""),
            config_hash=d.get("config_hash", ""),
        )


class VersionManager:
    """Manage configuration versions with snapshot, diff, and rollback.

    Each version is stored as a complete YAML copy of the config in a
    versioned subdirectory under versions_dir/.
    """

    def __init__(self, versions_dir: str = _DEFAULT_DIR):
        self._root = Path(versions_dir).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        config_dict: dict[str, Any],
        description: str = "",
        run_id: str = "",
    ) -> str:
        """Save a config snapshot as a new version.

        Args:
            config_dict: Full configuration dictionary (as loaded from YAML).
            description: Optional description of this version.
            run_id: Optional pipeline run ID.

        Returns:
            Version ID string, e.g. "v3".
        """
        version_id = self._next_id()
        ver_dir = self._root / version_id
        ver_dir.mkdir(parents=True, exist_ok=True)

        # Save config YAML
        config_path = ver_dir / _VERSION_FILENAME
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

        # Compute hash
        config_hash = self._hash(config_dict)

        # Update index
        meta = ConfigVersion(
            id=version_id,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            description=description,
            run_id=run_id,
            config_hash=config_hash,
        )
        self._upsert_meta(meta)

        logger.info("Config version %s saved%s",
                     version_id,
                     f" — {description}" if description else "")
        return version_id

    def list(self) -> list[ConfigVersion]:
        """List all saved versions, newest first."""
        index = self._load_index()
        return sorted(index.values(), key=lambda v: v.id, reverse=True)

    def show(self, version_id: str) -> dict[str, Any]:
        """Return the full config for a given version.

        Raises FileNotFoundError if the version does not exist.
        """
        config_path = self._root / version_id / _VERSION_FILENAME
        if not config_path.exists():
            raise FileNotFoundError(
                f"Version '{version_id}' not found at {config_path}"
            )
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def diff(self, v1: str, v2: str) -> str:
        """Unified diff between two config versions.

        Args:
            v1: First version ID (shown as "before").
            v2: Second version ID (shown as "after").

        Returns:
            Unified diff string, or empty string if identical.
        """
        cfg1 = self._flatten(self.show(v1))
        cfg2 = self._flatten(self.show(v2))

        lines1 = [f"{k}: {v}" for k, v in sorted(cfg1.items())]
        lines2 = [f"{k}: {v}" for k, v in sorted(cfg2.items())]

        diff = list(difflib.unified_diff(
            lines1, lines2,
            fromfile=f"config/{v1}",
            tofile=f"config/{v2}",
            lineterm="",
        ))
        return "\n".join(diff)

    def rollback(self, version_id: str, target_path: str | Path | None = None) -> None:
        """Restore a version's config to the active config file.

        Args:
            version_id: Version to restore (e.g. "v3").
            target_path: Path to write restored config. Default: config/default.yaml.
        """
        config = self.show(version_id)

        if target_path is None:
            target_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
        target_path = Path(target_path)

        # Backup current config before overwriting
        if target_path.exists():
            backup_dir = self._root / "_rollback_backups"
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / f"{target_path.name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(target_path, backup_path)
            logger.info("Current config backed up to %s", backup_path)

        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info("Config rolled back to version %s → %s", version_id, target_path)

    def delete(self, version_id: str) -> None:
        """Delete a specific version."""
        ver_dir = self._root / version_id
        if ver_dir.exists():
            shutil.rmtree(ver_dir)
        index = self._load_index()
        index.pop(version_id, None)
        self._save_index(index)
        logger.info("Deleted config version %s", version_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_id(self) -> str:
        """Generate the next version ID (v1, v2, ...)."""
        existing = [p.name for p in self._root.iterdir() if p.is_dir() and p.name.startswith("v")]
        numbers = []
        for name in existing:
            try:
                numbers.append(int(name[1:]))
            except (ValueError, IndexError):
                pass
        n = max(numbers) + 1 if numbers else 1
        return f"v{n}"

    def _load_index(self) -> dict[str, ConfigVersion]:
        """Load the version index from disk."""
        index_path = self._root / _META_FILENAME
        if not index_path.exists():
            return {}
        try:
            with open(index_path, encoding="utf-8") as f:
                raw = json.load(f)
            return {k: ConfigVersion.from_dict(v) for k, v in raw.items()}
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupted version index: %s — starting fresh", e)
            return {}

    def _save_index(self, index: dict[str, ConfigVersion]) -> None:
        """Save the version index to disk."""
        index_path = self._root / _META_FILENAME
        raw = {k: v.to_dict() for k, v in index.items()}
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)

    def _upsert_meta(self, meta: ConfigVersion) -> None:
        """Add or update a version in the index."""
        index = self._load_index()
        index[meta.id] = meta
        self._save_index(index)

    @staticmethod
    def _hash(config_dict: dict[str, Any]) -> str:
        """Compute a stable hash of a config dict."""
        import hashlib
        raw = json.dumps(config_dict, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten a nested dict into dot-separated keys for diffing."""
        result: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(VersionManager._flatten(v, key))
            elif isinstance(v, list):
                # For lists, show length and a summary
                if v and isinstance(v[0], dict):
                    result[key] = f"[{len(v)} items]"
                else:
                    result[key] = str(v)
            else:
                result[key] = v
        return result
