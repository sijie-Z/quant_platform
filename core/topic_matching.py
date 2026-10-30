"""Topic pattern matching shared by both event buses.

The two implementations disagreed: the legacy bus treated `market.*` as the
raw prefix `"market"` (so it matched `market` and `market_data.tick`), while
the async bus required an exact segment and never matched the documented
`order.**`. Subscriptions and history queries therefore returned different
sets for the same pattern.

Semantics:
- `**` or `*` alone       matches every topic
- `prefix.**`             matches `prefix` and every descendant
- `prefix.*`              matches exactly one additional segment
- anything else           exact match
"""

from __future__ import annotations


def topic_matches(topic: str, pattern: str) -> bool:
    if pattern in ("**", "*"):
        return True
    if pattern == topic:
        return True

    if pattern.endswith(".**"):
        prefix = pattern[:-3]
        return topic == prefix or topic.startswith(prefix + ".")

    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        if not topic.startswith(prefix + "."):
            return False
        return "." not in topic[len(prefix) + 1:]

    return False
