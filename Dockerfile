# ---------------------------------------------------------------------------
# Stage 1 — build the Vue dashboard.
# Building it in a Node image keeps node/npm out of the runtime image entirely
# (the previous single-stage build installed and then purged them via apt).
# ---------------------------------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /frontend

# Manifests first, so `npm ci` layers cache independently of source edits.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — runtime.
# Every dependency pinned in requirements.txt ships a manylinux wheel for
# CPython 3.12, so no compiler toolchain or OpenBLAS headers are needed. If a
# future pin has no wheel, the build will fail here loudly rather than
# silently pulling in a 500 MB apt toolchain.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# Cache mount keeps pip's wheel cache across builds, so a retry after a
# network hiccup only re-fetches the package that failed. The cache lives
# outside the image, so the image stays lean.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 5 --timeout 60 -r requirements.txt

# Application code
COPY . .

# quant_core first: `core/store.py` and friends are thin shims that re-export
# from quant_core, so without this the app dies at import time with
# `ModuleNotFoundError: No module named 'quant_core'`.
RUN pip install --no-cache-dir --no-deps ./packages/quant_core

# Editable install for CLI entry points and pip metadata
RUN pip install -e . --no-deps

# Prebuilt dashboard from stage 1
COPY --from=frontend /frontend/dist ./frontend/dist

# Symlink so `import quant_platform` resolves in any context:
# WORKDIR=/app, but the package is named quant_platform, so we make
# /quant_platform → /app available via PYTHONPATH=/.
RUN ln -sf /app /quant_platform
ENV PYTHONPATH=/:/app

# ---------------------------------------------------------------------------
# Drop privileges. Only the paths the application actually writes to are
# handed to the runtime user; the code itself stays root-owned and read-only
# to the process, so a compromised app cannot rewrite itself.
#
# The writable set was enumerated from the source, not guessed:
#   data/               research Registry (SQLite) + generated reports
#   results/            config: output.results_dir
#   .quant_cache/       PipelineCache
#   .cache/             baostock provider cache
#   .agent_cache/       ResearchAgent JSON cache
#   .sentiment_cache/   LLMSentimentFactor JSON cache
#   .quant_versions/    ConfigVersionManager snapshots
# Each one already exists with content or is created here, so the named volumes
# in docker-compose inherit the right ownership on first use.
# ---------------------------------------------------------------------------
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p data/reports results .quant_cache .cache .agent_cache .sentiment_cache .quant_versions \
    && chown -R appuser:appuser \
        data results .quant_cache .cache .agent_cache .sentiment_cache .quant_versions

USER appuser

EXPOSE 8000

# python:3.12-slim ships no curl/wget, so probe with the stdlib.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5).status == 200 else 1)"

CMD ["python", "main.py", "web"]
