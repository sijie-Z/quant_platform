"""Concurrent analysis reads share one context build (one analytics slot)."""

import asyncio

from quant_platform.api import research_context


def test_concurrent_builds_share_one_factory_call():
    calls = {"n": 0}

    async def run():
        async def factory():
            calls["n"] += 1
            await asyncio.sleep(0.05)
            return {"context": True}

        results = await asyncio.gather(
            research_context._single_flight("same-key", factory),
            research_context._single_flight("same-key", factory),
            research_context._single_flight("same-key", factory),
        )
        return results

    results = asyncio.run(run())
    assert calls["n"] == 1
    assert all(result == {"context": True} for result in results)


def test_failure_propagates_to_followers():
    async def run():
        async def factory():
            await asyncio.sleep(0.01)
            raise RuntimeError("boom")

        outcomes = await asyncio.gather(
            research_context._single_flight("fail-key", factory),
            research_context._single_flight("fail-key", factory),
            return_exceptions=True,
        )
        return outcomes

    outcomes = asyncio.run(run())
    assert all(isinstance(outcome, RuntimeError) for outcome in outcomes)
