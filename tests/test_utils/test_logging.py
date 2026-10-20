"""Logger naming must not double-prefix the package."""

from quant_platform.utils.logging import get_logger


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
