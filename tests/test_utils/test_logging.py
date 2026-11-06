"""Logger naming must not double-prefix the package."""

import logging
from logging.handlers import RotatingFileHandler

from quant_platform.utils.logging import get_logger, resolve_level, setup_logging


def test_module_name_is_not_double_prefixed():
    """`get_logger(__name__)` used to yield quant_platform.quant_platform.*.

    Every module passes its already-qualified __name__, so log lines showed
    the package twice and `logging.getLogger("quant_platform.data")` could
    not filter its own modules.
    """
    logger = get_logger("quant_platform.data.providers.synthetic")
    assert logger.name == "quant_platform.data.providers.synthetic"


def test_short_name_is_namespaced():
    assert get_logger("custom").name == "quant_platform.custom"
    assert get_logger("quant_platform").name == "quant_platform"


class TestLogLevelEnvironment:
    """S23: LOG_LEVEL from docker-compose was never read."""

    def test_log_level_env_is_used(self, monkeypatch):
        monkeypatch.delenv("QUANT_LOG_LEVEL", raising=False)
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        assert resolve_level() == logging.DEBUG

    def test_quant_log_level_wins_over_log_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("QUANT_LOG_LEVEL", "WARNING")
        assert resolve_level() == logging.WARNING

    def test_explicit_argument_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        assert resolve_level("ERROR") == logging.ERROR

    def test_unknown_level_falls_back_to_info(self, monkeypatch, capsys):
        monkeypatch.delenv("QUANT_LOG_LEVEL", raising=False)
        monkeypatch.setenv("LOG_LEVEL", "CHATTY")
        assert resolve_level() == logging.INFO
        assert "unknown log level" in capsys.readouterr().err

    def test_setup_logging_honors_the_environment(self, monkeypatch):
        monkeypatch.delenv("QUANT_LOG_LEVEL", raising=False)
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        logger = setup_logging()
        assert logger.level == logging.WARNING


class TestRotatingFileHandler:
    """S11: FileHandler log files grew without bound."""

    def test_file_output_rotates(self, tmp_path):
        log_file = tmp_path / "logs" / "quant.log"
        logger = setup_logging(
            level=logging.INFO, log_file=log_file,
            max_bytes=1024, backup_count=2,
        )

        rotating = [
            h for h in logger.handlers if isinstance(h, RotatingFileHandler)
        ]
        assert len(rotating) == 1
        assert rotating[0].maxBytes == 1024
        assert rotating[0].backupCount == 2

        # Rotation happens before a write that would exceed the limit, so
        # emit several records rather than one large one.
        for index in range(40):
            logger.info("line %d %s", index, "x" * 200)
        for handler in logger.handlers:
            handler.flush()

        # The file rolled over to a .1 backup.
        assert log_file.exists()
        assert (tmp_path / "logs" / "quant.log.1").exists()
