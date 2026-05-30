"""Tests for the Config Version Manager.

Inspired by wudao-hero-skill's version_manager.py, this module adds
explicit versioning, rollback, and diff to config management.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from quant_platform.utils.version_manager import (
    ConfigVersion,
    VersionManager,
)


@pytest.fixture
def tmp_versions_dir():
    """Create a temporary versions directory for testing."""
    tmp = Path(tempfile.mkdtemp(prefix="quant_versions_test_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def vm(tmp_versions_dir):
    """Return a VersionManager using the temp directory."""
    return VersionManager(str(tmp_versions_dir))


@pytest.fixture
def sample_config():
    """A sample config dict for testing."""
    return {
        "universe": {"n_stocks": 500, "exclude_st": True},
        "data": {
            "provider": "synthetic",
            "start_date": "2021-01-01",
            "end_date": "2025-12-31",
        },
        "alpha": {"method": "equal_weight", "lookback": 252},
        "portfolio": {
            "optimizer": "mean_variance",
            "constraints": {"max_weight": 0.05, "max_sector_exposure": 0.30},
        },
    }


# ---------------------------------------------------------------------------
# Test ConfigVersion dataclass
# ---------------------------------------------------------------------------


class TestConfigVersion:
    def test_to_dict_and_from_dict(self):
        v = ConfigVersion(
            id="v1",
            timestamp="2026-05-31T12:00:00",
            description="Initial config",
            run_id="run_001",
            config_hash="abc123",
        )
        d = v.to_dict()
        assert d["id"] == "v1"
        assert d["run_id"] == "run_001"

        restored = ConfigVersion.from_dict(d)
        assert restored.id == v.id
        assert restored.timestamp == v.timestamp
        assert restored.description == v.description
        assert restored.run_id == v.run_id
        assert restored.config_hash == v.config_hash

    def test_from_dict_minimal(self):
        d = {"id": "v2", "timestamp": "2026-01-01T00:00:00"}
        v = ConfigVersion.from_dict(d)
        assert v.id == "v2"
        assert v.description == ""
        assert v.run_id == ""


# ---------------------------------------------------------------------------
# Test VersionManager
# ---------------------------------------------------------------------------


class TestVersionManager:
    def test_init_creates_directory(self, tmp_versions_dir):
        dir_path = tmp_versions_dir / "subdir"
        vm = VersionManager(str(dir_path))
        assert dir_path.exists()
        assert dir_path.is_dir()

    def test_save_creates_version_file(self, vm, sample_config):
        vid = vm.save(sample_config, description="First version")
        assert vid.startswith("v")
        assert (vm._root / vid / "version.yaml").exists()

    def test_save_returns_incremental_ids(self, vm, sample_config):
        v1 = vm.save(sample_config)
        v2 = vm.save(sample_config)
        v3 = vm.save(sample_config)
        assert v1 == "v1"
        assert v2 == "v2"
        assert v3 == "v3"

    def test_saved_config_is_identical(self, vm, sample_config):
        vid = vm.save(sample_config)
        loaded = vm.show(vid)
        assert loaded == sample_config

    def test_list_returns_versions_newest_first(self, vm, sample_config):
        v1 = vm.save(sample_config, description="v1")
        v2 = vm.save(sample_config, description="v2")
        v3 = vm.save(sample_config, description="v3")
        versions = vm.list()

        assert len(versions) == 3
        assert versions[0].id == v3  # newest first
        assert versions[1].id == v2
        assert versions[2].id == v1

    def test_list_includes_metadata(self, vm, sample_config):
        vm.save(sample_config, description="Test run", run_id="abc123")
        versions = vm.list()
        assert len(versions) == 1
        assert versions[0].description == "Test run"
        assert versions[0].run_id == "abc123"

    def test_show_nonexistent_raises(self, vm):
        with pytest.raises(FileNotFoundError):
            vm.show("v999")

    def test_diff_identical(self, vm, sample_config):
        vm.save(sample_config)
        vm.save(sample_config)
        diff = vm.diff("v1", "v2")
        assert diff == ""

    def test_diff_different(self, vm, sample_config):
        vm.save(sample_config)
        modified = dict(sample_config)
        modified["alpha"]["method"] = "ic_weighted"
        vm.save(modified)
        diff = vm.diff("v1", "v2")

        assert diff != ""
        assert "ic_weighted" in diff
        assert "equal_weight" in diff

    def test_rollback_creates_file(self, vm, sample_config, tmp_versions_dir):
        vm.save(sample_config)

        # Modify config for v2
        modified = dict(sample_config)
        modified["alpha"]["method"] = "ic_weighted"
        vm.save(modified)

        # Rollback to v1
        target = tmp_versions_dir / "restored.yaml"
        vm.rollback("v1", target_path=target)
        assert target.exists()

        with open(target, encoding="utf-8") as f:
            restored = yaml.safe_load(f)
        assert restored["alpha"]["method"] == "equal_weight"

    def test_rollback_with_backup(self, vm, sample_config, tmp_versions_dir):
        """Rollback should create a backup of the current config."""
        vm.save(sample_config)
        modified = dict(sample_config)
        modified["alpha"]["method"] = "ic_weighted"
        vm.save(modified)

        target = tmp_versions_dir / "active.yaml"
        # Write a "current" config first
        with open(target, "w", encoding="utf-8") as f:
            yaml.dump(modified, f)

        vm.rollback("v1", target_path=target)
        # Backup should exist
        backup_dir = vm._root / "_rollback_backups"
        backups = list(backup_dir.glob("active.yaml.*"))
        assert len(backups) >= 1

    def test_delete_removes_version(self, vm, sample_config):
        vm.save(sample_config)
        vm.save(sample_config)
        assert len(vm.list()) == 2

        vm.delete("v1")
        assert len(vm.list()) == 1

        # v1 should not be showable
        with pytest.raises(FileNotFoundError):
            vm.show("v1")

    def test_empty_list(self, vm):
        versions = vm.list()
        assert versions == []

    def test_empty_config(self, vm):
        vid = vm.save({})
        assert vid == "v1"
        config = vm.show(vid)
        assert config == {}

    def test_save_with_run_id(self, vm, sample_config):
        vm.save(sample_config, run_id="run_xyz")
        versions = vm.list()
        assert versions[0].run_id == "run_xyz"

    def test_index_persistence(self, vm, sample_config):
        """Index should persist across VersionManager instances."""
        vm.save(sample_config, description="Persistent test")

        # Create a new VM pointing to the same directory
        vm2 = VersionManager(str(vm._root))
        versions = vm2.list()
        assert len(versions) == 1
        assert versions[0].description == "Persistent test"

    def test_rollback_nonexistent_version(self, vm):
        with pytest.raises(FileNotFoundError):
            vm.rollback("v999")

    def test_hash_changes_with_config(self, vm, sample_config):
        v1_id = vm.save(sample_config)
        v1 = vm.list()[0]

        modified = dict(sample_config)
        modified["portfolio"]["optimizer"] = "risk_parity"
        v2_id = vm.save(modified)
        v2 = vm.list()[0]  # v2 is newest

        assert v1.config_hash != v2.config_hash

    def test_flatten_nested_dict(self):
        from quant_platform.utils.version_manager import VersionManager
        d = {
            "a": {"b": 1, "c": 2},
            "d": 3,
            "e": {"f": {"g": 4}},
        }
        flat = VersionManager._flatten(d)
        assert flat["a.b"] == 1
        assert flat["a.c"] == 2
        assert flat["d"] == 3
        assert flat["e.f.g"] == 4


# ---------------------------------------------------------------------------
# Integration: auto-save on run
# ---------------------------------------------------------------------------


class TestAutoSave:
    """Verify the auto-save integration works end-to-end."""

    def test_run_saves_version(self, monkeypatch, vm, sample_config, tmp_versions_dir):
        """Simulate the auto-save that happens in cmd_run."""
        from quant_platform.utils.version_manager import VersionManager

        # Monkeypatch VersionManager to use temp dir
        orig_init = VersionManager.__init__
        monkeypatch.setattr(
            VersionManager, "__init__",
            lambda self, versions_dir=None: orig_init(self, versions_dir or str(tmp_versions_dir)),
        )

        # Should be usable immediately
        mgr = VersionManager()
        desc = f"Run: alpha=equal_weight optimizer=mean_variance"
        vid = mgr.save(sample_config, description=desc)
        assert vid == "v1"

        versions = mgr.list()
        assert len(versions) == 1
        assert "alpha=equal_weight" in versions[0].description

    def test_save_with_complex_config(self, vm):
        """Config with nested structures should save and restore correctly."""
        config = {
            "factors": {
                "technical": {
                    "momentum_1m": {"enabled": True, "params": {"period": 21}},
                    "volatility_20d": {"enabled": True, "params": {"period": 20}},
                },
                "processing": {
                    "winsorize": {"enabled": True, "lower": 0.01, "upper": 0.99},
                },
            },
            "instruments": [
                {"symbol": "600519", "asset_type": "stock", "multiplier": 1.0},
                {"symbol": "IF2406", "asset_type": "future", "multiplier": 300},
            ],
        }
        vid = vm.save(config, description="Complex config")
        restored = vm.show(vid)
        assert restored["factors"]["technical"]["momentum_1m"]["params"]["period"] == 21
        assert restored["instruments"][0]["symbol"] == "600519"
        assert restored["instruments"][1]["multiplier"] == 300
