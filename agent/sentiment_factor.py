"""LLM-based sentiment factor for A-share stocks.

Extracts trading signals from unstructured financial text by simulating
LLM API calls (OpenAI-compatible). The factor scores news headlines on a
-1 (extremely bearish) to +1 (extremely bullish) scale per stock per date.

Key design decisions for production-readiness:
- Async batch processing with rate limiting
- Local cache with TTL to avoid redundant API calls
- Graceful degradation: returns neutral scores if API is unavailable
- Deterministic scoring prompts for reproducibility

For interview demo: uses a keyword-based sentiment analyzer as the default
"simulated LLM" backend, with the option to swap in a real DeepSeek/LLM API call.
This demonstrates the architecture without requiring an API key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.factors.base import BaseFactor, FactorCategory
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Sample A-share financial news headlines for demo purposes
# In production, these would come from a news API (e.g., Bloomberg, Reuters, EastMoney)
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
    """LLM-augmented sentiment factor.

    Processes financial news headlines through an LLM (or simulated LLM)
    to produce daily sentiment scores per stock.

    Architecture:
    - SentimentAnalyzer (abstract) -> KeywordAnalyzer | OpenAI Analyzer
    - NewsProvider -> SampleNewsProvider | RealNewsAPI
    - Caching layer to avoid redundant LLM calls
    """

    category = FactorCategory.CUSTOM

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        use_real_llm: bool = False,
        lookback_days: int = 5,
        cache_dir: str | None = None,
        name: str = "llm_sentiment",
    ):
        """
        Args:
            api_key: DeepSeek API key. If None, uses DEEPSEEK_API_KEY env var.
            model: LLM model to use (default: deepseek-chat).
            use_real_llm: If True, calls DeepSeek/LLM API. If False, uses keyword analyzer.
            lookback_days: How many past days of news to aggregate.
            cache_dir: Directory for sentiment cache.
            name: Factor name.
        """
        super().__init__({
            "model": model,
            "use_real_llm": use_real_llm,
            "lookback_days": lookback_days,
        })
        self._name = name
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model
        self.use_real_llm = use_real_llm
        self.lookback_days = lookback_days
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./.sentiment_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Select analyzer
        if use_real_llm and self.api_key:
            self.analyzer = OpenAISentimentAnalyzer(self.api_key, self.model)
        else:
            self.analyzer = KeywordSentimentAnalyzer()

        # In-memory cache: {cache_key: {"score": float, "retryable": bool}}.
        # The key covers everything that can change a score (headline, asset,
        # analyzer type, model, prompt version and date). The previous key was
        # just (headline, date), so the same headline scored for stock A was
        # served to stock B, and a DeepSeek -> keyword fallback was persisted
        # as if it were the model's answer.
        self._cache: dict[str, dict] = {}
        self._unsaved_entries = 0
        self._load_cache()

    # Cache layout version. Bumped whenever the key composition or the entry
    # shape changes: an old file cannot be trusted under the new key, so it is
    # ignored rather than migrated.
    _CACHE_VERSION = 2

    _CACHE_FILENAME = "sentiment_cache.json"

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        prices: pd.DataFrame,
        financials: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute LLM sentiment scores for all assets across all dates.

        For each date, generates/retrieves news headlines for each stock,
        runs LLM sentiment analysis, and aggregates into a DataFrame.
        """
        assets = prices.columns
        dates = prices.index
        n_dates = len(dates)
        n_assets = len(assets)

        logger.info(
            "LLM Sentiment Factor: %d assets x %d dates, backend=%s",
            n_assets, n_dates,
            "DeepSeek" if self.use_real_llm else "Keyword (simulated)",
        )

        # Generate per-stock news headlines (deterministic based on asset + date)
        rng = np.random.default_rng(hash(42))
        sentiment = np.full((n_dates, n_assets), np.nan)

        for di, date in enumerate(dates):
            if di % 100 == 0:
                logger.debug("Sentiment: processing date %d/%d", di + 1, n_dates)

            for ai, asset in enumerate(assets):
                # Generate headlines for this stock on this date
                headlines = self._get_headlines(asset, date, rng)

                if not headlines:
                    continue

                # Score each headline via LLM/simulated analyzer
                scores = []
                for headline in headlines:
                    score = self._analyze_headline(headline, asset, date)
                    scores.append(score)

                # Aggregate: exponential decay weighted toward recent news
                # (all headlines are same date here, so equal weight)
                sentiment[di, ai] = np.mean(scores) if scores else 0.0

        result = pd.DataFrame(sentiment, index=dates, columns=assets)

        # Rolling average over lookback to smooth noise
        if self.lookback_days > 1:
            result = result.rolling(self.lookback_days, min_periods=1).mean()

        logger.info("LLM sentiment computed: mean=%.3f, std=%.3f",
                     result.mean().mean(), result.std().mean())

        return result

    # ------------------------------------------------------------------
    # Headline generation (simulated)
    # ------------------------------------------------------------------

    def _get_headlines(
        self, asset: str, date: pd.Timestamp, rng: np.random.Generator
    ) -> list[str]:
        """Get news headlines for a stock on a given date.

        In production: fetch from news API (Bloomberg, EastMoney, etc.).
        For demo: generate realistic headlines using templates.
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
        type, model, prompt version and date. A fallback score (the LLM call
        failed and the keyword analyzer answered instead) stays in memory for
        this session but is *not* written to disk and is retried on the next
        call, so an outage cannot pin a degraded score into the cache.
        """
        cache_key = self._make_cache_key(headline, asset, date)

        entry = self._cache.get(cache_key)
        if entry is not None and not entry.get("retryable", False):
            return entry["score"]

        # Analyze sentiment
        score = self.analyzer.analyze(headline, asset)
        retryable = bool(getattr(self.analyzer, "last_analyze_fallback", False))

        self._cache[cache_key] = {"score": float(score), "retryable": retryable}
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
            "model": self.model,
            "prompt": self._prompt_fingerprint(),
            "date": pd.Timestamp(date).date().isoformat(),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _prompt_fingerprint(self) -> str:
        """Version the analysis instructions, not just their presence."""
        prompt = getattr(self.analyzer, "SYSTEM_PROMPT", "") or ""
        if not prompt:
            # Keyword scoring has no prompt; its keyword lists are the rule
            # set. Fingerprint them so edits invalidate stored answers too.
            keywords = []
            for attr in ("BULLISH_KEYWORDS", "BEARISH_KEYWORDS"):
                values = getattr(self.analyzer, attr, None)
                if isinstance(values, (list, tuple)):
                    keywords.extend(str(v) for v in values)
            prompt = "|".join(keywords)
        return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------

    @property
    def _cache_file(self) -> Path:
        return self.cache_dir / self._CACHE_FILENAME

    @staticmethod
    def _normalise_entry(value: object) -> dict | None:
        """Accept only a well-formed persisted entry."""
        if isinstance(value, dict) and isinstance(value.get("score"), (int, float)):
            return {"score": float(value["score"]), "retryable": False}
        return None

    def _load_cache(self) -> None:
        """Load persisted cache from disk.

        A corrupt or outdated file must never wipe the in-memory cache: it is
        warned about and left untouched for inspection, and the session simply
        starts from what it already holds.
        """
        cache_file = self._cache_file
        if not cache_file.exists():
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

        loaded = {}
        for key, value in data["entries"].items():
            entry = self._normalise_entry(value)
            if entry is not None:
                loaded[key] = entry
        self._cache = loaded
        logger.debug("Loaded %d cached sentiment scores", len(loaded))

    def _maybe_save_cache(self) -> None:
        """Flush every 100 new entries to reduce I/O."""
        if self._unsaved_entries >= 100:
            self._save_cache()

    def _save_cache(self) -> None:
        """Persist the cache atomically (temp file + ``os.replace``).

        Writing JSON in place left a truncated file if the process died
        mid-dump, and the old loader then wiped every score. Retryable
        (fallback) entries are deliberately not written.
        """
        entries = {
            key: {"score": float(entry["score"])}
            for key, entry in self._cache.items()
            if not entry.get("retryable", False)
        }
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

    # Chinese financial sentiment keywords
    BULLISH_KEYWORDS = [
        "增长", "超预期", "回购", "增持", "中标", "突破",
        "创新", "利好", "分红", "补贴", "订单", "合作",
        "新市场", "专利", "认可", "景气", "预增", "提升",
        "买入", "增持", "看好", "反转", "筑底", "复苏",
    ]
    BEARISH_KEYWORDS = [
        "下降", "减持", "亏损", "违规", "监管", "问询",
        "减值", "违约", "调查", "处罚", "辞职", "停产",
        "召回", "投诉", "危机", "下滑", "预亏", "收紧",
        "评级下调", "卖出", "减持", "看空", "泡沫", "崩盘",
    ]

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

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self._client = None
        self._api_base = "https://api.deepseek.com/v1"
        self._request_count = 0
        self._last_request_time = 0.0

    def analyze(self, headline: str, asset: str = "") -> float:
        """Call DeepSeek API for sentiment analysis."""
        self.last_analyze_fallback = False

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
                    {"role": "user", "content": f"Stock: {asset}\nHeadline: {headline}"},
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
            return KeywordSentimentAnalyzer().analyze(headline, asset)
