"""Configuration version manager.

Inspired by wudao-hero-skill's version_manager.py, this module adds
explicit versioning, rollback, and diff capabilities to configuration
management. Each version is a full snapshot of the active config.

Versions are stored as YAML files in a versions directory with a
JSON metadata index.

Concurrency contract (issue S13): version-id allocation, the version
directory creation and the index update run under an exclusive cross-process
file lock; the id is reserved with an atomic ``mkdir`` (O_EXCL semantics) and
the index is written temp-file + ``os.replace``. Two processes (or threads)
saving at the same time therefore never lose a version or overwrite an
index entry.

Secret hygiene: secret-shaped keys (``execution.qmt.password``, tokens, API
keys, ...) are replaced with ``***REDACTED***`` before anything is written to
disk or returned by :meth:`show`/:meth:`diff`. Older snapshots that still
contain a plaintext secret are redacted on read as well.

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
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

_VERSION_FILENAME = "version.yaml"
_META_FILENAME = "index.json"
_LOCK_FILENAME = ".versions.lock"
_DEFAULT_DIR = ".quant_versions"
REDACTED = "***REDACTED***"

#: Keys whose values must never be persisted in plaintext. Matched
#: case-insensitively against the mapping key, at any nesting depth.
_SECRET_KEYS = frozenset({
    "password", "passwd", "pass", "secret", "token", "api_key", "apikey",
    "api_secret", "access_key", "access_token", "private_key",
    "secret_key", "client_secret", "auth_token", "credential",
})

#: Substring patterns for secret-shaped keys. Exact matching missed
#: ``tushare_token``, ``deepseek_api_key`` and ``postgres_dsn``, which were
#: written to ``.quant_versions/vN/version.yaml`` in plaintext.
_SECRET_KEY_PATTERNS = (
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "api-key", "access_key", "private_key", "credential", "dsn",
)

#: ``scheme://user:password@host/...`` -- a URL with embedded credentials.
_CREDENTIAL_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://[^/\s@]+@")


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _SECRET_KEYS:
        return True
    if any(pattern in lowered for pattern in _SECRET_KEY_PATTERNS):
        return True
    return lowered == "auth" or lowered.endswith("_auth")


_SENSITIVE_QUERY_KEYS = (
    "token", "key", "secret", "password", "passwd", "signature",
    "credential", "auth",
)


def _is_credential_url(value: str) -> bool:
    """Whether a URL carries credentials or a secret query parameter.

    Covers ``scheme://user:pass@host`` as well as webhook-style URLs such as
    ``https://alerts.example/hook?token=...`` that leaked verbatim before.
    """
    if _CREDENTIAL_URL_RE.match(value):
        return True
    if "://" not in value:
        return False
    try:
        from urllib.parse import parse_qsl, urlsplit

        pairs = parse_qsl(urlsplit(value).query, keep_blank_values=True)
    except ValueError:
        return False
    return any(
        any(pattern in key.lower() for pattern in _SENSITIVE_QUERY_KEYS)
        for key, _ in pairs
    )


class _FileLock:
    """Exclusive cross-process lock built on an O_EXCL lock file.

    ``msvcrt.locking`` semantics differ between platforms and ``fcntl`` is
    POSIX-only; creating the lock file with ``O_CREAT|O_EXCL`` is atomic on
    both Windows and POSIX and also serializes threads in one process. A lock
    older than ``stale_after`` seconds is treated as left behind by a crashed
    process and removed.
    """

    def __init__(self, path: Path, timeout: float = 10.0, stale_after: float = 120.0):
        self._path = path
        self._timeout = timeout
        self._stale_after = stale_after

    def __enter__(self) -> _FileLock:
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fd = os.open(
                    str(self._path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                try:
                    os.write(fd, str(os.getpid()).encode("ascii"))
                finally:
                    os.close(fd)
                return self
            except FileExistsError:
                try:
                    age = time.time() - self._path.stat().st_mtime
                    if age > self._stale_after:
                        self._path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for config-version lock {self._path}"
                    ) from None
                time.sleep(0.005)

    def __exit__(self, *exc_info) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass


def redact_secrets(value: Any) -> Any:
    """Return a deep copy with secret-shaped values replaced by REDACTED.

    Empty values are left alone: there is nothing to leak, and an empty
    string keeps the snapshot a faithful record of "not configured".
    """
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            non_empty = item not in (None, "", [], {}, ())
            if (
                isinstance(key, str)
                and non_empty
                and _is_secret_key(key)
            ):
                # Secret-shaped keys are redacted regardless of value shape:
                # ``api_tokens: ["a", "b"]`` used to recurse into the list and
                # leak every element because list items have no key.
                out[key] = REDACTED
            elif (
                isinstance(key, str)
                and isinstance(item, str)
                and item != ""
                and _is_credential_url(item)
            ):
                out[key] = REDACTED
            else:
                out[key] = redact_secrets(item)
        return out
    if isinstance(value, list):
        return [
            REDACTED if isinstance(item, str) and _is_credential_url(item)
            else redact_secrets(item)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    return value


def _strip_redacted(value: Any) -> Any:
    """Drop leaves whose value is the REDACTED placeholder.

    Used by rollback so restoring a snapshot that was stored with redacted
    secrets does not write the literal placeholder into the active config;
    the key is omitted and the env-first ``QMT_PASSWORD`` override applies.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            cleaned = _strip_redacted(item)
            if cleaned is REDACTED:
                continue
            out[key] = cleaned
        return out
    if isinstance(value, list):
        return [item for item in (_strip_redacted(v) for v in value) if item is not REDACTED]
    return value


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
        with _FileLock(self._root / _LOCK_FILENAME):
            while True:
                version_id = self._next_id()
                ver_dir = self._root / version_id
                try:
                    # Atomic reservation: two processes that computed the same
                    # id cannot both create the directory.
                    os.mkdir(ver_dir)
                    break
                except FileExistsError:
                    continue

            redacted = redact_secrets(config_dict)
            config_hash = self._hash(redacted)
            self._write_yaml(ver_dir / _VERSION_FILENAME, redacted)

            # Update index atomically inside the same lock.
            meta = ConfigVersion(
                id=version_id,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                description=description,
                run_id=run_id,
                config_hash=config_hash,
            )
            index = self._load_index()
            index[meta.id] = meta
            self._save_index(index)

        logger.info("Config version %s saved%s",
                    version_id,
                    f" — {description}" if description else "")
        return version_id

    def list(self) -> list[ConfigVersion]:
        """List all saved versions, newest first."""
        index = self._load_index()
        return sorted(index.values(), key=lambda v: v.id, reverse=True)

    def show(self, version_id: str) -> dict[str, Any]:
        """Return the full config for a given version (secrets redacted).

        Raises FileNotFoundError if the version does not exist.
        """
        config_path = self._root / version_id / _VERSION_FILENAME
        if not config_path.exists():
            raise FileNotFoundError(
                f"Version '{version_id}' not found at {config_path}"
            )
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # Old snapshots may predate redaction; never print them verbatim.
        return redact_secrets(raw)

    def diff(self, v1: str, v2: str) -> str:
        """Unified diff between two config versions (secrets redacted).

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
        config = _strip_redacted(self.show(version_id))

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

        self._write_yaml(target_path, config)
        logger.info("Config rolled back to version %s → %s", version_id, target_path)

    def delete(self, version_id: str) -> None:
        """Delete a specific version."""
        with _FileLock(self._root / _LOCK_FILENAME):
            self._delete_locked(version_id)
        logger.info("Deleted config version %s", version_id)

    def prune(self, keep: int = 20, max_age_days: float | None = None) -> list[str]:
        """Delete old versions by count and/or age. Returns removed ids.

        Args:
            keep: Retain at least the ``keep`` newest versions (clamped to 1).
            max_age_days: Also delete versions older than this many days.
        """
        keep = max(1, int(keep))
        removed: list[str] = []
        with _FileLock(self._root / _LOCK_FILENAME):
            versions = self._load_index()
            ordered = sorted(versions.values(), key=self._version_number)
            if max_age_days is not None:
                cutoff = datetime.now() - timedelta(days=float(max_age_days))
                survivors = []
                for version in ordered:
                    try:
                        created = datetime.fromisoformat(version.timestamp)
                    except ValueError:
                        created = None
                    if created is not None and created < cutoff:
                        self._delete_locked(version.id)
                        removed.append(version.id)
                    else:
                        survivors.append(version)
                ordered = survivors
            overflow = len(ordered) - keep
            if overflow > 0:
                for version in ordered[:overflow]:
                    self._delete_locked(version.id)
                    removed.append(version.id)
        if removed:
            logger.info("Pruned %d config version(s): %s", len(removed), ", ".join(removed))
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _delete_locked(self, version_id: str) -> None:
        ver_dir = self._root / version_id
        if ver_dir.exists():
            shutil.rmtree(ver_dir, ignore_errors=True)
        index = self._load_index()
        if index.pop(version_id, None) is not None:
            self._save_index(index)

    def _next_id(self) -> str:
        """Generate the next version ID (v1, v2, ...)."""
        existing = [
            p.name for p in self._root.iterdir()
            if p.is_dir() and p.name.startswith("v")
        ]
        numbers = []
        for name in existing:
            try:
                numbers.append(int(name[1:]))
            except (ValueError, IndexError):
                pass
        n = max(numbers) + 1 if numbers else 1
        return f"v{n}"

    @staticmethod
    def _version_number(version: ConfigVersion) -> int:
        try:
            return int(version.id[1:])
        except (ValueError, IndexError):
            return 0

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
        """Save the version index atomically (temp file + os.replace)."""
        index_path = self._root / _META_FILENAME
        raw = {k: v.to_dict() for k, v in index.items()}
        self._write_text_atomic(
            index_path,
            json.dumps(raw, indent=2, ensure_ascii=False),
        )

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @classmethod
    def _write_yaml(cls, path: Path, data: Any) -> None:
        cls._write_text_atomic(
            path,
            yaml.dump(data, default_flow_style=False, allow_unicode=True),
        )

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
