"""Guard tests for nested config coercion.

``config/schema.py`` declares nested dataclasses (``DataConfig.synthetic``,
``PortfolioConfig.constraints``, ...).  ``load_config()`` used to build only a
hand-written list of them; anything not on that list was either left as the raw
YAML ``dict`` (``data.synthetic``) or never read at all, so the section silently
took its dataclass defaults.

These tests enumerate ``config/schema.py`` itself rather than listing sections,
so a nested config added later is covered without anyone remembering to add a
case here.
"""

import dataclasses
import types as _types
import typing

import pytest

import quant_platform.config.schema as schema
from quant_platform.utils.config import load_config

# `factors` is not a plain mirror of its YAML section: load_config() flattens
# `factors.technical.<name>.enabled` / `factors.fundamental.<name>.enabled`
# into the `enabled_technicals` / `enabled_fundamentals` tuples, so its raw
# section deliberately does not round-trip.
_SECTIONS_NOT_ROUND_TRIPPED = {"factors"}


def _nested_dataclass(field_type):
    """The dataclass inside the declared type, or None.

    Matches a bare dataclass and ``list[SomeDataclass]`` / ``tuple[...]``.
    """
    if dataclasses.is_dataclass(field_type):
        return field_type
    if typing.get_origin(field_type) in (list, tuple):
        for arg in typing.get_args(field_type):
            if arg is not Ellipsis and dataclasses.is_dataclass(arg):
                return arg
    return None


def _declared_nested_paths(cls, prefix=""):
    """Yield (dotted_path, nested_dataclass) for every nested dataclass field."""
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        nested = _nested_dataclass(hints[f.name])
        if nested is None:
            continue
        path = f"{prefix}{f.name}"
        yield path, nested
        if typing.get_origin(hints[f.name]) not in (list, tuple):
            yield from _declared_nested_paths(nested, prefix=f"{path}.")


def _resolve(obj, dotted_path):
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def _field_default(f):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()
    return None


def _sentinel(field_type, default):
    """A raw value of the declared type, distinguishable from ``default``."""
    origin = typing.get_origin(field_type)

    # Optional[X] / X | None -> use X
    if origin is typing.Union or origin is getattr(_types, "UnionType", None):
        inner = [a for a in typing.get_args(field_type) if a is not type(None)]
        return _sentinel(inner[0], default) if inner else None

    if dataclasses.is_dataclass(field_type):
        hints = typing.get_type_hints(field_type)
        return {f.name: _sentinel(hints[f.name], _field_default(f))
                for f in dataclasses.fields(field_type)}

    if origin in (list, tuple):
        args = [a for a in typing.get_args(field_type) if a is not Ellipsis]
        item = _sentinel(args[0] if args else str, None)
        return [item] if origin is list else (item,)

    if field_type is bool:
        return not (default if isinstance(default, bool) else False)
    if field_type is int:
        return (default if isinstance(default, int) else 0) + 987654
    if field_type is float:
        return (default if isinstance(default, float) else 0.0) + 0.987654
    if field_type is str:
        return f"{default}__sentinel__" if isinstance(default, str) else "__sentinel__"
    return "__sentinel__"


def _build_sentinel_raw():
    """One raw YAML-shaped dict for the whole Config, every field sentineled."""
    raw = {}
    for f in dataclasses.fields(schema.Config):
        if f.name in _SECTIONS_NOT_ROUND_TRIPPED:
            continue
        hints = typing.get_type_hints(schema.Config)
        raw[f.name] = _sentinel(hints[f.name], _field_default(f))
    return raw


def _diff(cls, loaded, raw, prefix, problems):
    """Compare a loaded dataclass against the raw mapping it came from."""
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        path = f"{prefix}{f.name}"
        if path in _SECTIONS_NOT_ROUND_TRIPPED:
            continue

        got = getattr(loaded, f.name)
        want = raw[f.name]
        nested = _nested_dataclass(hints[f.name])

        if nested is None:
            if got != want:
                problems.append(f"{path}: yaml={want!r} loaded={got!r}")
            continue

        if typing.get_origin(hints[f.name]) in (list, tuple):
            items = list(got) if isinstance(got, (list, tuple)) else [got]
            if len(items) != len(want):
                problems.append(f"{path}: yaml has {len(want)}, loaded has {len(items)}")
                continue
            for i, (item, item_raw) in enumerate(zip(items, want, strict=False)):
                if not isinstance(item, nested):
                    problems.append(
                        f"{path}[{i}]: loaded {type(item).__name__}, declared {nested.__name__}")
                    continue
                _diff(nested, item, item_raw, f"{path}[{i}].", problems)
        else:
            if not isinstance(got, nested):
                problems.append(
                    f"{path}: loaded {type(got).__name__}, declared {nested.__name__}")
                continue
            _diff(nested, got, want, f"{path}.", problems)


class TestNestedConfigIsCoerced:
    """No declared nested config may come back as a raw dict."""

    def test_every_declared_nested_config_is_the_declared_type(self):
        """The guard for the raw-dict hole.

        A nested section left as a raw ``dict`` makes every
        ``getattr(cfg.<section>, '<field>', <fallback>)`` in the codebase return
        its fallback, so the config is silently ignored.
        """
        cfg = load_config()

        offenders = []
        for path, declared in _declared_nested_paths(schema.Config):
            value = _resolve(cfg, path)
            items = value if isinstance(value, (list, tuple)) else [value]
            if not items:
                continue  # empty list of a declared element type is fine
            for item in items:
                if not isinstance(item, declared):
                    offenders.append(
                        f"{path} -> {type(item).__name__} (declared {declared.__name__})")
                    break

        assert not offenders, (
            "load_config() left declared nested config as a raw dict: "
            + "; ".join(offenders)
        )

    def test_every_declared_field_of_every_section_round_trips(self):
        """The value-level guard.

        ``test_every_declared_nested_config_is_the_declared_type`` only checks
        types, and a section that is never read from YAML still has the right
        type because the dataclass default_factory produces one. This asserts
        the YAML content actually arrives.
        """
        raw = _build_sentinel_raw()
        cfg = load_config(raw=raw)

        problems = []
        _diff(schema.Config, cfg, raw, "", problems)

        assert not problems, (
            "YAML values did not reach the loaded Config: " + "; ".join(problems)
        )

    @pytest.mark.parametrize(
        "path",
        [p for p, _ in _declared_nested_paths(schema.Config)],
    )
    def test_declared_path_resolves_on_the_default_config(self, path):
        """Each declared path exists and is reachable on the shipped default."""
        _resolve(load_config(), path)
