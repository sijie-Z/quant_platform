"""LLM-based sentiment factor for A-share stocks.

Extracts trading signals from unstructured financial text by scoring news
headlines on a -1 (extremely bearish) to +1 (extremely bullish) scale per
stock per date.

Data sources (explicit, because it decides what the score means):
- ``compute(..., news={asset: {date: [headline, ...]}})`` scores *caller
  supplied* headlines. This is the only path that produces real news
  sentiment, and it is the only path usable with a real LLM backend unless
  the caller explicitly opts into the synthetic demo.
- With no ``news`` argument the factor falls back to a built-in demo
  generator that fabricates headlines from templates with an RNG. Those
  scores are flagged ``is_synthetic=True`` (also stored in ``params`` and
  set as ``data_source="synthetic_templates"``) and must not be used as
  alpha or presented as news sentiment. There is no bundled news API
  client: plug your provider in and pass the headlines.

Backend honesty:
- ``use_real_llm=True`` without an API key is downgraded to the keyword
  analyzer, and the object says so: ``requested_use_real_llm`` is True while
  ``use_real_llm``/``is_llm`` are False and ``backend`` is ``"keyword"``.
  Logs print the *actual* backend, never the requested one.
- With a real LLM backend, ``compute()`` refuses to run without
  caller-supplied news (``allow_synthetic_llm=True`` opts into an explicitly
  labeled synthetic demo) so fabricated templates are never silently sent to
  a paid model and reported as LLM sentiment.

Other production-readiness decisions:
- Async-friendly batch processing with rate limiting
- Local cache with a real TTL (per-entry timestamps; expired entries are
  re-scored) and strict validation of persisted entries (finite, within
  [-1, 1], versioned) so a corrupt cache cannot inject scores
- Graceful degradation: keyword fallback scores are retried, never persisted
- Deterministic scoring prompts for reproducibility
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.factors.base import BaseFactor, FactorCategory
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Sample A-share financial news headlines for the DEMO path only.
# These are fabricated from templates with an RNG; they are not news and the
# resulting scores are flagged synthetic by LLMSentimentFactor.compute().
SAMPLE_HEADLINES_POOL = [
    # Bullish signals
    ("净利润同比增长{:.0f}%", 0.6),
    ("发布{:.1f}亿元回购计划", 0.5),
    ("获得{major_order}大额订单", 0.5),
    ("新产品{product}获市场认可", 0.4),
    ("机构上调目标价至{:.1f}元", 0.4),
    ("年报业绩超预期", 0.5),
    ("签订战略合作协议", 0.3),
    ("获得政府补贴{:.1f}亿元", 0.2),
    ("控股股东增持", 0.4),
    ("行业景气度回升", 0.3),
    ("中标重大项目", 0.4),
    ("技术突破获专利", 0.3),
    ("海外市场拓展顺利", 0.3),
    ("分红比例提升", 0.3),
    ("业绩预告大幅预增", 0.5),

    # Bearish signals
    ("净利润同比下降{:.0f}%", -0.5),
    ("大股东减持{:.0f}万股", -0.4),
    ("收到监管问询函", -0.5),
    ("商誉减值{:.1f}亿元", -0.6),
    ("被立案调查", -0.7),
    ("业绩预告预亏", -0.5),
    ("债务违约风险", -0.6),
    ("高管辞职", -0.3),
    ("产品质量问题召回", -0.4),
    ("行业政策收紧", -0.3),
    ("遭客户投诉索赔", -0.4),
    ("子公司经营困难", -0.3),
    ("大额对外担保风险", -0.4),
    ("信用评级下调", -0.5),
    ("原材料价格大幅上涨", -0.3),

    # Neutral / mixed signals
    ("公布定期报告", 0.0),
    ("召开股东大会", 0.0),
    ("变更会计师事务所", -0.1),
    ("董事会换届选举", 0.0),
    ("公司章程修订", 0.0),
    ("披露投资者关系活动记录", 0.1),
    ("发布澄清公告", -0.1),
    ("停牌筹划重大事项", 0.0),
    ("完成工商变更登记", 0.0),
    ("设立子公司", 0.1),
]


class LLMSentimentFactor(BaseFactor):
    """Sentiment factor with a pluggable keyword or real-LLM analyzer.

    Data sources:
    - Caller-supplied headlines via ``compute(..., news=...)`` — real news
      sentiment (``is_synthetic=False``).
    - Built-in demo generator (no ``news`` argument) — fabricated template
      headlines, ``is_synthetic=True``, not fit for alpha.

    Architecture:
    - SentimentAnalyzer (abstract) -> KeywordSentimentAnalyzer |
      OpenAISentimentAnalyzer
    - News is supplied by the caller; this module ships no news API client.
    - Caching layer with per-entry TTL and validated, versioned entries.
    """

    category = FactorCategory.CUSTOM

    #: Maximum characters of a single headline accepted/scored. Longer input
    #: is truncated (and logged) so a hostile or broken upstream feed cannot
    #: push unbounded text into an LLM prompt or the cache key.
    MAX_HEADLINE_CHARS = 500

    #: Maximum headlines scored per asset-day bucket. Extras are dropped with
    #: a warning; the cap keeps LLM request volume bounded.
    MAX_HEADLINES_PER_ASSET_DAY = 20

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        use_real_llm: bool = False,
        lookback_days: int = 5,
        cache_dir: str | None = None,
        name: str = "llm_sentiment",
        cache_ttl_days: float | None = 7.0,
        allow_synthetic_llm: bool = False,
        max_cache_entries: int = 100_000,
        max_cache_file_bytes: int = 64 * 1024 * 1024,
    ):
        """
        Args:
            api_key: DeepSeek API key. If None, uses DEEPSEEK_API_KEY env var.
            model: LLM model to use (default: deepseek-chat).
            use_real_llm: If True, requests the DeepSeek LLM backend. Without
                an API key the factor is *explicitly downgraded* to the
                keyword analyzer (``requested_use_real_llm`` True,
                ``use_real_llm`` False, ``backend`` "keyword") and a warning
                is logged.
            lookback_days: How many past days of news to aggregate.
            cache_dir: Directory for sentiment cache.
            name: Factor name.
            cache_ttl_days: Age in days after which a cached score is
                re-computed. ``None`` disables expiry.
            allow_synthetic_llm: If True, a real-LLM backend may score the
                fabricated template headlines (explicit demo only; output
                stays flagged ``is_synthetic=True``). Default False: the
                real-LLM path requires caller-supplied ``news``.
            max_cache_entries: In-memory cache size cap; the oldest ~10% of
                entries are evicted when it is reached.
            max_cache_file_bytes: Refuse to load a cache file larger than
                this (guards against a poisoned/oversized file).
        """
        if int(lookback_days) != lookback_days or lookback_days < 1:
            raise ValueError("lookback_days must be an integer >= 1")
        if cache_ttl_days is not None and not float(cache_ttl_days) > 0:
            raise ValueError("cache_ttl_days must be positive or None")
        if int(max_cache_entries) != max_cache_entries or max_cache_entries < 1:
            raise ValueError("max_cache_entries must be an integer >= 1")
        if max_cache_file_bytes < 1:
            raise ValueError("max_cache_file_bytes must be positive")

        super().__init__({
            "model": model,
            "requested_use_real_llm": bool(use_real_llm),
            "lookback_days": int(lookback_days),
            "cache_ttl_days": cache_ttl_days,
        })
        self._name = name
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model
        self.lookback_days = int(lookback_days)
        self.cache_ttl_days = cache_ttl_days
        self.cache_ttl_seconds = (
            None if cache_ttl_days is None else float(cache_ttl_days) * 86400.0
        )
        self.max_cache_entries = int(max_cache_entries)
        self.max_cache_file_bytes = int(max_cache_file_bytes)
        self._allow_synthetic_llm = bool(allow_synthetic_llm)
        # Constructor params are immutable provenance; per-run provenance is
        # built into a *new* dict in compute() so an earlier FactorResult is
        # never retroactively relabeled by a later run.
        self._base_params = dict(self.params)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./.sentiment_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # What the caller asked for vs. what will actually run. Only the
        # effective backend is exposed through ``use_real_llm``/``backend``;
        # the request is preserved separately so the downgrade is visible.
        self.requested_use_real_llm = bool(use_real_llm)
        if use_real_llm and self.api_key:
            self.analyzer = OpenAISentimentAnalyzer(self.api_key, self.model)
        else:
            self.analyzer = KeywordSentimentAnalyzer()
        if self.backend_downgraded:
            logger.warning(
                "use_real_llm=True requested but no API key is available "
                "(DEEPSEEK_API_KEY unset); downgraded to KeywordSentimentAnalyzer: "
                "backend=%s, is_llm=False. This run will NOT produce LLM output.",
                self.backend,
            )

        # Provenance for the last compute() call. Defaults describe the
        # no-news demo path; compute() updates them.
        self.is_synthetic: bool = True
        self.data_source: str = "synthetic_templates"

        # Run-level analyzer outcome. ``n_fallback_scores`` counts calls that
        # silently degraded (LLM error -> keyword); a real-LLM run with
        # fallback scores is partly keyword output and says so in params/logs.
        self.n_analyzer_calls = 0
        self.n_fallback_scores = 0
        self.last_fallback_reason: str | None = None

        # In-memory cache: {cache_key: {"score": float, "ts": float,
        # "retryable": bool}}. The key covers everything that can change a
        # score (headline, asset, analyzer type, model, prompt version and
        # date). The previous key was just (headline, date), so the same
        # headline scored for stock A was served to stock B, and a
        # DeepSeek -> keyword fallback was persisted as if it were the
        # model's answer.
        self._cache: dict[str, dict] = {}
        self._unsaved_entries = 0
        self._load_cache()

    # Cache layout version. Bumped whenever the key composition or the entry
    # shape changes: an old file cannot be trusted under the new key, so it is
    # ignored rather than migrated. Version 4 folds the user-message template
    # and the live analyzer model into the key (version 3 added TTL/entry
    # versions).
    _CACHE_VERSION = 4

    _CACHE_FILENAME = "sentiment_cache.json"

    #: Clock-skew tolerance for persisted entry timestamps (seconds). An entry
    #: stamped in the future beyond this window is treated as invalid.
    _MAX_CLOCK_SKEW = 86400.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_llm(self) -> bool:
        """True only when the *actual* analyzer is the real LLM client."""
        return isinstance(self.analyzer, OpenAISentimentAnalyzer)

    @property
    def use_real_llm(self) -> bool:
        """Effective backend flag (never the requested-but-downgraded one)."""
        return self.is_llm

    @property
    def backend(self) -> str:
        """Machine-readable label for the analyzer that actually runs."""
        if isinstance(self.analyzer, OpenAISentimentAnalyzer):
            return f"llm:{self.model}"
        if isinstance(self.analyzer, KeywordSentimentAnalyzer):
            return "keyword"
        return type(self.analyzer).__name__

    @property
    def backend_downgraded(self) -> bool:
        """True when a real-LLM backend was requested but keyword runs.

        Derived from the live analyzer, so it stays truthful even if the
        analyzer object is replaced after construction.
        """
        return self.requested_use_real_llm and not self.is_llm

    def compute(
        self,
        prices: pd.DataFrame,
        financials: pd.DataFrame | None = None,
        news: Mapping | None = None,
        headlines: Mapping | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute sentiment scores for all assets across all dates.

        Args:
            prices: Price panel (date x asset). Only the index/columns are
                used; values do not enter the score.
            financials: Ignored (factor signature compatibility).
            news: Caller-supplied headlines, shaped as
                ``{asset: {date-like: [headline, ...]}}`` (a single string
                per date is also accepted). Dates are matched to the price
                index after normalisation; asset keys are matched by their
                string form. Days without supplied headlines are missing
                (NaN) before smoothing; with ``lookback_days > 1`` the rolling
                mean can carry a supplied score forward for a few days. No
                template headlines are fabricated for missing days. Passing
                this makes ``is_synthetic`` False.
            headlines: Alias for ``news`` (pass exactly one).

        Returns:
            DataFrame (date x asset) of sentiment scores in [-1, 1].

        Raises:
            ValueError: if both ``news`` and ``headlines`` are given, if a
                real-LLM backend was selected without caller-supplied news
                (unless ``allow_synthetic_llm=True``), or for invalid
                configuration values.
            TypeError: if ``news`` is not a mapping.
        """
        if news is not None and headlines is not None:
            raise ValueError("pass either news= or headlines=, not both")
        supplied = news if news is not None else headlines

        self.n_analyzer_calls = 0
        self.n_fallback_scores = 0
        self.last_fallback_reason = None

        if (
            supplied is None
            and self.is_llm
            and not self._allow_synthetic_llm
        ):
            raise ValueError(
                "LLMSentimentFactor: the real-LLM backend is active but no "
                "caller-supplied news was passed. Refusing to send fabricated "
                "template headlines to the model. Pass "
                "news={asset: {date: [headline, ...]}} or construct with "
                "allow_synthetic_llm=True for an explicitly labeled synthetic demo."
            )

        assets = prices.columns
        dates = prices.index
        n_dates = len(dates)
        n_assets = len(assets)

        prepared: dict[tuple[object, pd.Timestamp], list[str]] | None
        if supplied is None:
            self.is_synthetic = True
            self.data_source = "synthetic_templates"
            prepared = None
            logger.warning(
                "LLMSentimentFactor demo mode: no caller-supplied news, so "
                "headlines are generated from %d templates with an RNG. Scores "
                "are SYNTHETIC (is_synthetic=True, data_source=synthetic_templates) "
                "and must not be used as alpha or reported as news sentiment. "
                "Pass news={asset: {date: [headline, ...]}} for real data.",
                len(SAMPLE_HEADLINES_POOL),
            )
        else:
            self.is_synthetic = False
            self.data_source = "caller_supplied_news"
            prepared = self._prepare_news(supplied, assets, dates)
            logger.info(
                "LLMSentimentFactor using caller-supplied news: %d asset-day "
                "bucket(s) across %d asset(s)",
                len(prepared), n_assets,
            )

        logger.info(
            "LLM Sentiment Factor: %d assets x %d dates, backend=%s "
            "(requested=%s, synthetic=%s, data_source=%s)",
            n_assets, n_dates,
            self.backend,
            "llm" if self.requested_use_real_llm else "keyword",
            self.is_synthetic,
            self.data_source,
        )

        sentiment = np.full((n_dates, n_assets), np.nan)

        if prepared is not None:
            prepared_news = prepared
            for di, date in enumerate(dates):
                if di % 100 == 0:
                    logger.debug("Sentiment: processing date %d/%d", di + 1, n_dates)
                key_date = self._normalise_date(date)
                for ai, asset in enumerate(assets):
                    headlines = prepared_news.get((asset, key_date), [])
                    if not headlines:
                        continue
                    scores = [
                        self._analyze_headline(headline, asset, date)
                        for headline in headlines
                    ]
                    sentiment[di, ai] = np.mean(scores) if scores else 0.0
        else:
            # Demo path: deterministic per-asset RNG drives template choice.
            rng = np.random.default_rng(hash(42))
            for di, date in enumerate(dates):
                if di % 100 == 0:
                    logger.debug("Sentiment: processing date %d/%d", di + 1, n_dates)
                for ai, asset in enumerate(assets):
                    headlines = self._get_headlines(asset, date, rng)
                    if not headlines:
                        continue
                    scores = [
                        self._analyze_headline(headline, asset, date)
                        for headline in headlines
                    ]
                    sentiment[di, ai] = np.mean(scores) if scores else 0.0

        result = pd.DataFrame(sentiment, index=dates, columns=assets)

        # Rolling average over lookback to smooth noise
        if self.lookback_days > 1:
            result = result.rolling(self.lookback_days, min_periods=1).mean()

        if self.n_fallback_scores:
            logger.warning(
                "Sentiment run partially degraded: %d of %d analyzer call(s) "
                "fell back to keywords (backend=%s). Those scores are keyword "
                "output, not LLM output; reason=%s",
                self.n_fallback_scores, self.n_analyzer_calls,
                self.backend, self.last_fallback_reason,
            )

        self.params = {
            **self._base_params,
            "requested_use_real_llm": self.requested_use_real_llm,
            "effective_backend": self.backend,
            "is_synthetic": self.is_synthetic,
            "data_source": self.data_source,
            "n_analyzer_calls": self.n_analyzer_calls,
            "n_fallback_scores": self.n_fallback_scores,
            "fallback_reason": self.last_fallback_reason,
        }

        logger.info(
            "Sentiment computed: mean=%.3f, std=%.3f (backend=%s, synthetic=%s, "
            "fallback_scores=%d/%d)",
            result.mean().mean(), result.std().mean(),
            self.backend, self.is_synthetic,
            self.n_fallback_scores, self.n_analyzer_calls,
        )

        return result

    # ------------------------------------------------------------------
    # Caller-supplied news ingestion
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_date(value: object) -> pd.Timestamp | None:
        """Parse a date-like value to a naive, normalised Timestamp."""
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(stamp):
            return None
        if stamp.tzinfo is not None:
            stamp = stamp.tz_localize(None)
        return stamp.normalize()

    def _prepare_news(
        self,
        news: object,
        assets: Iterable[object],
        dates: Iterable[object],
    ) -> dict[tuple[object, pd.Timestamp], list[str]]:
        """Validate and index caller-supplied headlines.

        Unknown assets/dates and malformed headline entries are warned about
        and skipped; oversized input is truncated to keep prompts bounded.
        """
        if not isinstance(news, Mapping):
            raise TypeError(
                "news must be a mapping {asset: {date: [headline, ...]}}; "
                f"got {type(news).__name__}"
            )

        asset_lookup: dict[str, object] = {str(asset): asset for asset in assets}
        price_dates = {
            key for key in (self._normalise_date(date) for date in dates)
            if key is not None
        }
        prepared: dict[tuple[object, pd.Timestamp], list[str]] = {}
        skipped_assets = 0
        skipped_dates = 0
        skipped_headlines = 0

        for asset_key, per_date in news.items():
            asset = asset_lookup.get(str(asset_key))
            if asset is None:
                skipped_assets += 1
                logger.warning(
                    "news: asset %r is not in the price panel; ignoring its headlines",
                    asset_key,
                )
                continue
            if not isinstance(per_date, Mapping):
                skipped_dates += 1
                logger.warning(
                    "news[%r] must map date-like keys to headlines; got %s",
                    asset_key, type(per_date).__name__,
                )
                continue

            for date_key, raw_headlines in per_date.items():
                key_date = self._normalise_date(date_key)
                if key_date is None or key_date not in price_dates:
                    skipped_dates += 1
                    logger.warning(
                        "news[%r]: date %r is not in the price index; ignored",
                        asset_key, date_key,
                    )
                    continue

                if isinstance(raw_headlines, str):
                    raw_headlines = [raw_headlines]
                if isinstance(raw_headlines, Mapping) or not isinstance(
                    raw_headlines, Iterable
                ):
                    skipped_headlines += 1
                    logger.warning(
                        "news[%r][%r] must be a string or list of strings; got %s",
                        asset_key, date_key, type(raw_headlines).__name__,
                    )
                    continue

                cleaned: list[str] = []
                for headline in raw_headlines:
                    if not isinstance(headline, str):
                        skipped_headlines += 1
                        logger.warning(
                            "news[%r][%r]: non-string headline %r ignored",
                            asset_key, date_key, headline,
                        )
                        continue
                    text = headline.strip()
                    if not text:
                        continue
                    if len(text) > self.MAX_HEADLINE_CHARS:
                        logger.warning(
                            "news[%r][%r]: headline truncated from %d to %d chars",
                            asset_key, date_key, len(text), self.MAX_HEADLINE_CHARS,
                        )
                        text = text[: self.MAX_HEADLINE_CHARS]
                    cleaned.append(text)

                if len(cleaned) > self.MAX_HEADLINES_PER_ASSET_DAY:
                    logger.warning(
                        "news[%r][%r]: %d headlines supplied, keeping the first %d",
                        asset_key, date_key, len(cleaned),
                        self.MAX_HEADLINES_PER_ASSET_DAY,
                    )
                    cleaned = cleaned[: self.MAX_HEADLINES_PER_ASSET_DAY]

                if cleaned:
                    bucket_key = (asset, key_date)
                    merged = prepared.get(bucket_key, []) + cleaned
                    if len(merged) > self.MAX_HEADLINES_PER_ASSET_DAY:
                        logger.warning(
                            "news[%r][%r]: %d headlines after merging duplicate "
                            "date keys, keeping the first %d",
                            asset_key, date_key, len(merged),
                            self.MAX_HEADLINES_PER_ASSET_DAY,
                        )
                        merged = merged[: self.MAX_HEADLINES_PER_ASSET_DAY]
                    prepared[bucket_key] = merged

        if skipped_assets or skipped_dates or skipped_headlines:
            logger.warning(
                "news ingestion skipped %d unknown asset(s), %d date entr(ies) "
                "and %d malformed headline(s)",
                skipped_assets, skipped_dates, skipped_headlines,
            )
        return prepared

    # ------------------------------------------------------------------
    # Headline generation (explicitly synthetic demo path)
    # ------------------------------------------------------------------

    def _get_headlines(
        self, asset: str, date: pd.Timestamp, rng: np.random.Generator
    ) -> list[str]:
        """Generate DEMO headlines for a stock on a given date.

        These headlines are fabricated from templates with an RNG — they are
        not news. ``compute()`` labels any score produced from them
        ``is_synthetic=True``. Real data must be passed via ``news=``.
        """
        # Each stock has a ~15% chance of having news on any given day
        if rng.random() > 0.15:
            return []

        n_headlines = rng.integers(1, 3)  # 1-2 headlines per day
        headlines = []

        for _ in range(n_headlines):
            template, base_score = SAMPLE_HEADLINES_POOL[
                rng.integers(0, len(SAMPLE_HEADLINES_POOL))
            ]
            # Fill in template with random numbers
            headline = self._fill_template(template, rng)
            headlines.append(headline)

        return headlines

    def _fill_template(self, template: str, rng: np.random.Generator) -> str:
        """Fill template placeholders with random values."""
        # Replace {:.Nf} format specs
        result = template
        for match in re.finditer(r"\{[^}]*\}", template):
            spec = match.group()
            if ":.0f" in spec:
                val = rng.integers(10, 200)
                result = result.replace(spec, str(val), 1)
            elif ":.1f" in spec:
                val = rng.uniform(0.5, 50)
                result = result.replace(spec, f"{val:.1f}", 1)
            elif "{major_order}" in spec:
                result = result.replace("{major_order}",
                    rng.choice(["海外", "新能源", "基建", "军工"]), 1)
            elif "{product}" in spec:
                result = result.replace("{product}",
                    rng.choice(["芯片", "电池", "药物", "机器人"]), 1)

        return result

    # ------------------------------------------------------------------
    # Sentiment analysis with caching
    # ------------------------------------------------------------------

    def _analyze_headline(self, headline: str, asset: str, date: pd.Timestamp) -> float:
        """Analyze sentiment of a single headline.

        Results are cached under a key covering the headline, asset, analyzer
        type, model, prompt version and date. Entries expire after
        ``cache_ttl_days``. A fallback score (the LLM call failed and the
        keyword analyzer answered instead) stays in memory for this session
        but is *not* written to disk and is retried on the next call, so an
        outage cannot pin a degraded score into the cache.
        """
        headline = str(headline)[: self.MAX_HEADLINE_CHARS]
        cache_key = self._make_cache_key(headline, asset, date)
        now = time.time()

        entry = self._cache.get(cache_key)
        if entry is not None:
            if isinstance(entry, Mapping) and not entry.get("retryable", False):
                cached_score = self._valid_score(entry.get("score"))
                if cached_score is not None and not self._entry_expired(entry, now):
                    return cached_score
            # Invalid, expired or malformed: drop it and re-score.
            self._cache.pop(cache_key, None)

        # Analyze sentiment
        self.n_analyzer_calls += 1
        raw_score = self.analyzer.analyze(headline, asset)
        retryable = bool(getattr(self.analyzer, "last_analyze_fallback", False))
        score = self._valid_score(raw_score)
        fallback_reason: str | None = None
        if score is None:
            logger.warning(
                "Analyzer %s returned an invalid score %r for headline %r; "
                "recording 0.0 and leaving the entry retryable",
                type(self.analyzer).__name__, raw_score, headline[:80],
            )
            score = 0.0
            retryable = True
            fallback_reason = (
                f"analyzer {type(self.analyzer).__name__} returned an invalid "
                f"score ({raw_score!r})"
            )
        elif retryable:
            fallback_reason = (
                getattr(self.analyzer, "last_fallback_error", None)
                or f"{type(self.analyzer).__name__} fell back to keywords"
            )
        if fallback_reason is not None:
            self.n_fallback_scores += 1
            self.last_fallback_reason = fallback_reason

        self._maybe_evict_cache()
        self._cache[cache_key] = {"score": score, "ts": now, "retryable": retryable}
        if not retryable:
            self._unsaved_entries += 1
            self._maybe_save_cache()

        return score

    def _make_cache_key(
        self, headline: str, asset: str, date: pd.Timestamp
    ) -> str:
        """Build a cache key that distinguishes everything affecting a score.

        Two different assets sharing a headline must not share a score (the
        LLM prompt carries the stock), nor may a keyword fallback be replayed
        under the real model's name, nor may a prompt change silently reuse
        answers computed under the old instructions.
        """
        payload = {
            "headline": headline,
            "asset": str(asset),
            "analyzer": type(self.analyzer).__name__,
            # The live analyzer's model, not just the factor's configured one:
            # swapping in an analyzer trained on a different model must not
            # reuse the old model's answers.
            "model": str(getattr(self.analyzer, "model", self.model)),
            "prompt": self._prompt_fingerprint(),
            "date": pd.Timestamp(date).date().isoformat(),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _prompt_fingerprint(self) -> str:
        """Version the analysis instructions, not just their presence."""
        prompt = getattr(self.analyzer, "SYSTEM_PROMPT", "") or ""
        user_template = getattr(self.analyzer, "USER_PROMPT_TEMPLATE", "") or ""
        if not prompt:
            # Keyword scoring has no prompt; its keyword lists are the rule
            # set. Fingerprint them so edits invalidate stored answers too.
            keywords = []
            for attr in ("BULLISH_KEYWORDS", "BEARISH_KEYWORDS"):
                values = getattr(self.analyzer, attr, None)
                if isinstance(values, (list, tuple)):
                    keywords.extend(str(v) for v in values)
            prompt = "|".join(keywords)
        # The user-message template is part of the instructions too.
        fingerprint_source = f"{prompt}|user:{user_template}"
        return hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Score / entry validation
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_score(value: object) -> float | None:
        """Return the score as a float if it is finite and within [-1, 1]."""
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            return None
        score = float(value)
        if not math.isfinite(score) or not -1.0 <= score <= 1.0:
            return None
        return score

    @staticmethod
    def _valid_timestamp(value: object) -> float | None:
        """Return the timestamp as a float epoch if it looks like one."""
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            return None
        stamp = float(value)
        if not math.isfinite(stamp) or stamp <= 0:
            return None
        return stamp

    def _entry_expired(self, entry: Mapping, now: float | None = None) -> bool:
        """True when an in-memory cache entry is past its TTL."""
        if self.cache_ttl_seconds is None:
            return False
        stamp = self._valid_timestamp(entry.get("ts"))
        if stamp is None:
            return True
        now = time.time() if now is None else now
        if stamp > now + self._MAX_CLOCK_SKEW:
            return True
        return now - stamp > self.cache_ttl_seconds

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------

    @property
    def _cache_file(self) -> Path:
        return self.cache_dir / self._CACHE_FILENAME

    def _normalise_entry(self, value: object, now: float | None = None) -> dict | None:
        """Accept only a well-formed, versioned, unexpired persisted entry."""
        if not isinstance(value, dict):
            return None
        if value.get("version") != self._CACHE_VERSION:
            return None
        # A persisted retryable/fallback entry must never be promoted to a
        # trusted answer (fallbacks are supposed to stay in memory only).
        if value.get("retryable"):
            return None
        score = self._valid_score(value.get("score"))
        stamp = self._valid_timestamp(value.get("ts"))
        if score is None or stamp is None:
            return None
        now = time.time() if now is None else now
        if stamp > now + self._MAX_CLOCK_SKEW:
            return None
        if self.cache_ttl_seconds is not None and now - stamp > self.cache_ttl_seconds:
            return None
        return {"score": score, "ts": stamp, "retryable": False}

    def _load_cache(self) -> None:
        """Load persisted cache from disk.

        A corrupt or outdated file must never wipe the in-memory cache: it is
        warned about and left untouched for inspection, and the session simply
        starts from what it already holds. Individual entries are validated
        (version, finite score in [-1, 1], timestamp, TTL); invalid or expired
        entries are dropped with a warning instead of being served.
        """
        cache_file = self._cache_file
        if not cache_file.exists():
            return
        try:
            file_size = cache_file.stat().st_size
        except OSError as e:
            logger.warning("Sentiment cache %s is not readable (%s); ignored", cache_file, e)
            return
        if file_size > self.max_cache_file_bytes:
            logger.warning(
                "Sentiment cache %s is %d bytes, above the %d-byte limit; "
                "ignoring it (in-memory cache kept)",
                cache_file, file_size, self.max_cache_file_bytes,
            )
            return
        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(
                "Sentiment cache %s is unreadable (%s); keeping the in-memory "
                "cache and leaving the file in place",
                cache_file, e,
            )
            return

        if (
            not isinstance(data, dict)
            or data.get("version") != self._CACHE_VERSION
            or not isinstance(data.get("entries"), dict)
        ):
            logger.warning(
                "Sentiment cache %s has version/layout %r, expected %d; "
                "ignoring it (in-memory cache kept)",
                cache_file,
                data.get("version") if isinstance(data, dict) else type(data).__name__,
                self._CACHE_VERSION,
            )
            return

        now = time.time()
        loaded = {}
        dropped = 0
        for key, value in data["entries"].items():
            entry = self._normalise_entry(value, now=now)
            if not isinstance(key, str) or entry is None:
                dropped += 1
                continue
            loaded[key] = entry
        self._cache = loaded
        if len(loaded) > self.max_cache_entries:
            overflow = len(loaded) - self.max_cache_entries
            for key in self._oldest_keys(loaded, overflow):
                loaded.pop(key, None)
            logger.warning(
                "Sentiment cache %s: loaded %d entries; kept the %d newest",
                cache_file, len(loaded) + overflow, len(loaded),
            )
        if dropped:
            logger.warning(
                "Sentiment cache %s: ignored %d invalid or expired entr(ies)",
                cache_file, dropped,
            )
        logger.debug("Loaded %d cached sentiment scores", len(loaded))

    def _maybe_save_cache(self) -> None:
        """Flush every 100 new entries to reduce I/O."""
        if self._unsaved_entries >= 100:
            self._save_cache()

    def _oldest_keys(self, cache: Mapping, count: int) -> list[str]:
        """Return the ``count`` cache keys with the oldest (or missing) ts."""
        if count <= 0:
            return []

        def _sort_key(key: str) -> float:
            entry = cache.get(key)
            if not isinstance(entry, Mapping):
                return 0.0
            stamp = self._valid_timestamp(entry.get("ts"))
            return 0.0 if stamp is None else stamp

        return sorted(cache, key=_sort_key)[:count]

    def _maybe_evict_cache(self) -> None:
        """Keep the in-memory cache bounded, evicting the oldest ~10%."""
        if len(self._cache) < self.max_cache_entries:
            return
        target = max(1, self.max_cache_entries // 10)
        oldest = self._oldest_keys(self._cache, target)
        for key in oldest:
            self._cache.pop(key, None)
        logger.warning(
            "Sentiment cache reached %d entries; evicted the %d oldest",
            self.max_cache_entries, len(oldest),
        )

    def _save_cache(self) -> None:
        """Persist the cache atomically (temp file + ``os.replace``).

        Writing JSON in place left a truncated file if the process died
        mid-dump, and the old loader then wiped every score. Retryable
        (fallback) entries are deliberately not written, and entries whose
        score is not finite / within [-1, 1] are rejected instead of being
        persisted, so a corrupted in-memory value cannot poison the file.
        """
        now = time.time()
        entries = {}
        dropped = 0
        for key, entry in self._cache.items():
            if not isinstance(key, str) or not isinstance(entry, Mapping):
                dropped += 1
                continue
            if entry.get("retryable", False):
                continue
            score = self._valid_score(entry.get("score"))
            if score is None:
                dropped += 1
                continue
            stamp = self._valid_timestamp(entry.get("ts"))
            if stamp is None or stamp > now + self._MAX_CLOCK_SKEW:
                # Repair a missing/absurd timestamp to "now".
                stamp = now
            elif self.cache_ttl_seconds is not None and now - stamp > self.cache_ttl_seconds:
                # Expired scores are not worth persisting; they are re-scored
                # on the next run either way.
                continue
            entries[key] = {
                "score": score,
                "ts": stamp,
                "version": self._CACHE_VERSION,
            }
        if dropped:
            logger.warning(
                "Sentiment cache: refusing to persist %d invalid entr(ies)", dropped,
            )
        if len(entries) > self.max_cache_entries:
            overflow = len(entries) - self.max_cache_entries
            for key in self._oldest_keys(entries, overflow):
                entries.pop(key, None)
            logger.warning(
                "Sentiment cache: keeping the %d newest of %d entries",
                self.max_cache_entries, self.max_cache_entries + overflow,
            )
        payload = {"version": self._CACHE_VERSION, "entries": entries}
        cache_file = self._cache_file

        tmp_name = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f"{cache_file.name}.", suffix=".tmp", dir=self.cache_dir,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
                written = os.fstat(f.fileno()).st_size
            if written > self.max_cache_file_bytes:
                # Never write a file our own loader would refuse to read.
                raise ValueError(
                    f"serialized sentiment cache is {written} bytes, above the "
                    f"{self.max_cache_file_bytes}-byte limit; keeping the old file"
                )
            os.replace(tmp_name, cache_file)
            tmp_name = None
            self._unsaved_entries = 0
        except Exception as e:
            logger.warning("Failed to save sentiment cache: %s", e)
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass


# ======================================================================
# Sentiment Analyzers (Strategy pattern)
# ======================================================================

class SentimentAnalyzer:
    """Abstract sentiment analyzer."""

    # Set by an analyzer whose last call silently degraded (e.g. the LLM API
    # failed and a keyword score stood in for it). The factor uses this to keep
    # fallback scores out of the persistent cache.
    last_analyze_fallback: bool = False
    # Human-readable reason for the last fallback (exception text).
    last_fallback_error: str | None = None

    def analyze(self, headline: str, asset: str = "") -> float:
        """Return sentiment score: -1 (bearish) to +1 (bullish)."""
        raise NotImplementedError


class KeywordSentimentAnalyzer(SentimentAnalyzer):
    """Keyword/rule-based sentiment analyzer.

    Uses Chinese financial keyword dictionaries to score headlines.
    This is the DEFAULT backend — fast, deterministic, no API cost.

    In a real system, this serves as:
    - A baseline to compare against the LLM
    - A fallback when the LLM API is unavailable
    - A pre-filter to only send ambiguous headlines to the LLM
    """

    # Chinese financial sentiment keywords. De-duplicated on construction so a
    # headline matching one keyword is counted once: the old lists contained
    # "增持" (bullish) and "减持" (bearish) twice, so a mixed headline scored
    # 0.2 instead of 1/3.
    BULLISH_KEYWORDS = list(dict.fromkeys([
        "增长", "超预期", "回购", "增持", "中标", "突破",
        "创新", "利好", "分红", "补贴", "订单", "合作",
        "新市场", "专利", "认可", "景气", "预增", "提升",
        "买入", "看好", "反转", "筑底", "复苏",
    ]))
    BEARISH_KEYWORDS = list(dict.fromkeys([
        "下降", "减持", "亏损", "违规", "监管", "问询",
        "减值", "违约", "调查", "处罚", "辞职", "停产",
        "召回", "投诉", "危机", "下滑", "预亏", "收紧",
        "评级下调", "卖出", "看空", "泡沫", "崩盘",
    ]))

    def analyze(self, headline: str, asset: str = "") -> float:
        """Score headline by counting bullish vs bearish keywords."""
        bullish_count = sum(
            1 for kw in self.BULLISH_KEYWORDS if kw in headline
        )
        bearish_count = sum(
            1 for kw in self.BEARISH_KEYWORDS if kw in headline
        )

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        # Normalize to [-1, 1]
        raw = (bullish_count - bearish_count) / total
        return round(raw, 4)


class OpenAISentimentAnalyzer(SentimentAnalyzer):
    """DeepSeek/LLM API sentiment analyzer (OpenAI-compatible).

    Sends headlines to DeepSeek-Chat with a structured prompt asking for
    sentiment scores. Uses OpenAI-compatible API; any compatible provider
    (DeepSeek, Moonshot, etc.) works. Default: DeepSeek-Chat.
    """

    SYSTEM_PROMPT = """You are a financial sentiment analyst for Chinese A-share stocks.
Analyze the given news headline and output ONLY a single number between -1.0 and 1.0:
- 1.0: extremely bullish (strong positive catalyst)
- 0.5: moderately bullish
- 0.0: neutral / mixed
- -0.5: moderately bearish
- -1.0: extremely bearish (strong negative catalyst)

Output format: just the number, e.g. "0.6" or "-0.3". Nothing else."""

    #: User-message template; part of the prompt fingerprint.
    USER_PROMPT_TEMPLATE = "Stock: {asset}\nHeadline: {headline}"

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self._client = None
        self._api_base = "https://api.deepseek.com/v1"
        self._request_count = 0
        self._last_request_time = 0.0
        self.last_analyze_fallback = False
        self.last_fallback_error: str | None = None

    def analyze(self, headline: str, asset: str = "") -> float:
        """Call DeepSeek API for sentiment analysis."""
        self.last_analyze_fallback = False
        self.last_fallback_error = None
        # Defensive caps: prompts must stay bounded even if the analyzer is
        # called directly (outside LLMSentimentFactor's own validation).
        headline = str(headline)[:500]
        asset = str(asset)[:100]

        # Rate limiting: max 50 requests per second
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < 0.02:  # 50 req/s
            time.sleep(0.02 - elapsed)

        try:
            from openai import OpenAI
            if self._client is None:
                self._client = OpenAI(api_key=self.api_key, base_url=self._api_base)

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(
                        asset=asset, headline=headline,
                    )},
                ],
                max_tokens=10,
                temperature=0.0,  # Deterministic
            )

            self._last_request_time = time.monotonic()
            self._request_count += 1

            text = response.choices[0].message.content.strip()
            # Parse the number
            score = float(text)
            return max(-1.0, min(1.0, score))  # Clamp

        except Exception as e:
            logger.warning("OpenAI API call failed: %s, falling back to keyword analyzer", e)
            self.last_analyze_fallback = True
            self.last_fallback_error = str(e)
            return KeywordSentimentAnalyzer().analyze(headline, asset)
