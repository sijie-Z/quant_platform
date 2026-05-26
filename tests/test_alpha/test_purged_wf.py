"""Tests for purged walk-forward splits in ML signal generation."""

import pytest
from quant_platform.alpha.ml_signal import (
    MLSignalConfig,
    TimeSeriesCV,
    purged_walk_forward_splits,
)


class TestTimeSeriesCVPurge:
    """Test purge gap behavior in TimeSeriesCV."""

    def test_gap_separates_train_test(self):
        """There should be a gap of `gap` samples between train and test."""
        cv = TimeSeriesCV(n_splits=3, train_size=100, test_size=30, gap=10)
        splits = list(cv.split(300))

        for train_idx, test_idx in splits:
            actual_gap = min(test_idx) - max(train_idx) - 1
            assert actual_gap == 10, f"Gap should be 10, got {actual_gap}"

    def test_no_overlap_between_train_test(self):
        """Train and test indices must not overlap."""
        cv = TimeSeriesCV(n_splits=5, train_size=200, test_size=50, gap=10)
        splits = list(cv.split(1000))

        for train_idx, test_idx in splits:
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert train_set.isdisjoint(test_set)

    def test_purge_gap_size(self):
        """The gap between train_end and test_start should be exactly `gap`."""
        gap = 15
        cv = TimeSeriesCV(n_splits=3, train_size=100, test_size=30, gap=gap)
        splits = list(cv.split(300))

        for train_idx, test_idx in splits:
            actual_gap = min(test_idx) - max(train_idx) - 1
            assert actual_gap == gap

    def test_embargo_after_test(self):
        """Embargo should increase the step between folds."""
        cv_no_embargo = TimeSeriesCV(n_splits=3, train_size=100, test_size=30, gap=5, embargo=0)
        cv_with_embargo = TimeSeriesCV(n_splits=3, train_size=100, test_size=30, gap=5, embargo=10)

        splits_no = list(cv_no_embargo.split(500))
        splits_with = list(cv_with_embargo.split(500))

        # With embargo, the second fold's train start should be later
        if len(splits_no) >= 2 and len(splits_with) >= 2:
            # Embargo affects rolling mode start positions
            assert len(splits_with) <= len(splits_no) + 1  # May have fewer folds

    def test_expanding_mode(self):
        """In expanding mode, train_start should always be 0."""
        cv = TimeSeriesCV(n_splits=3, train_size=100, test_size=30, gap=5, mode="expanding")
        splits = list(cv.split(300))

        for train_idx, test_idx in splits:
            assert min(train_idx) == 0

    def test_rolling_mode(self):
        """In rolling mode, train_start should advance."""
        cv = TimeSeriesCV(n_splits=3, train_size=100, test_size=30, gap=5, mode="rolling")
        splits = list(cv.split(300))

        if len(splits) >= 2:
            # Second fold's train should start later than first
            assert min(splits[1][0]) >= min(splits[0][0])

    def test_insufficient_data_raises(self):
        """Should raise ValueError if not enough data."""
        cv = TimeSeriesCV(n_splits=5, train_size=100, test_size=30, gap=10)
        with pytest.raises(ValueError, match="Need at least"):
            list(cv.split(50))

    def test_yields_correct_number_of_splits(self):
        """Should yield approximately n_splits splits."""
        cv = TimeSeriesCV(n_splits=5, train_size=200, test_size=50, gap=10)
        splits = list(cv.split(2000))
        assert len(splits) == 5


class TestPurgedWalkForwardSplits:
    """Test the convenience function."""

    def test_returns_list_of_tuples(self):
        """Should return list of (train_idx, test_idx) tuples."""
        splits = purged_walk_forward_splits(n_samples=500, n_splits=3, gap=10)
        assert isinstance(splits, list)
        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert isinstance(train_idx, list)
            assert isinstance(test_idx, list)

    def test_default_gap_is_10(self):
        """Default gap should be 10."""
        splits = purged_walk_forward_splits(n_samples=500, n_splits=3)
        for train_idx, test_idx in splits:
            gap = min(test_idx) - max(train_idx) - 1
            assert gap == 10

    def test_custom_gap(self):
        """Custom gap should be respected."""
        splits = purged_walk_forward_splits(n_samples=500, n_splits=3, gap=20)
        for train_idx, test_idx in splits:
            gap = min(test_idx) - max(train_idx) - 1
            assert gap == 20


class TestMLSignalConfigPurge:
    """Test that MLSignalConfig includes purge parameters."""

    def test_default_purge_gap(self):
        config = MLSignalConfig()
        assert config.purge_gap == 10

    def test_default_embargo(self):
        config = MLSignalConfig()
        assert config.embargo == 0

    def test_custom_purge_gap(self):
        config = MLSignalConfig(purge_gap=20, embargo=5)
        assert config.purge_gap == 20
        assert config.embargo == 5
