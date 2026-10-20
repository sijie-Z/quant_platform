"""FastAPI application for the A-Share Multi-Factor Quant Platform.

Usage:
    python app.py                    # Start web server (default port 8000)
    python app.py --port 8080        # Custom port
    python app.py --no-frontend      # API only, no static files

    python main.py run               # CLI mode (original)
    python main.py web               # Same as python app.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the parent directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from quant_platform.api.monitor import router as monitor_router
from quant_platform.api.routes import router as api_router

# Determine frontend dist path
_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_DIST_DIR = _FRONTEND_DIR / "dist"


class APITokenMiddleware(BaseHTTPMiddleware):
    """Require a token (or loopback) for every non-health API route.

    When no token is configured, EVERY /api route is loopback-only, not just
    the trading/OMS subset: the shipped default (`main.py web` binds 0.0.0.0)
    let remote anonymous clients read orders, trades, signals, audit rows and
    trigger every compute endpoint. Health stays public for probes.
    """

    PUBLIC_PATHS = {
        "/api/health",
    }
    DOC_PATHS = {
        "/docs", "/redoc", "/docs/oauth2-redirect", "/openapi.json",
        "/api/docs", "/api/redoc",
    }
    LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost", "testclient")

    async def dispatch(self, request: Request, call_next):
        # Let CORS answer preflight; otherwise the browser sees a 401 without
        # CORS headers and the documented dev flow cannot use a token.
        if request.method == "OPTIONS":
            return await call_next(request)

        token = os.environ.get("QUANT_API_TOKEN", "")
        path = request.url.path
        protected = (
            (path.startswith("/api") and path not in self.PUBLIC_PATHS)
            or path in self.DOC_PATHS
        )
        if protected:
            if token:
                auth = request.headers.get("Authorization", "")
                api_key = request.headers.get("X-API-Key", "")
                if api_key != token and auth != f"Bearer {token}":
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API token"},
                    )
            else:
                host = request.client.host if request.client else ""
                if host not in self.LOOPBACK_HOSTS:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "detail": (
                                "No QUANT_API_TOKEN configured: all API routes "
                                "are loopback-only. Set the token to allow "
                                "remote access."
                            )
                        },
                    )
        return await call_next(request)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Run the event bus for the lifetime of the application.

    The bus is a process-wide singleton, but nothing ever started it. Without
    consumer loops every publish fell back to inline synchronous delivery,
    async handlers were skipped, and dead-letter retry never ran.
    """
    from quant_platform.core.events import get_event_bus

    bus = get_event_bus()
    await bus.start()
    try:
        yield
    finally:
        await bus.stop()


def _json_safe(value):
    """Replace non-finite floats so Starlette's strict encoder accepts them."""
    import math

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def create_app(serve_frontend: bool = True) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="A-Share Multi-Factor Quant Platform",
        description="Multi-factor quantitative research platform with LLM-enhanced stock selection",
        version="1.0.0",
        lifespan=_lifespan,
        docs_url="/api/docs" if serve_frontend else "/docs",
        redoc_url="/api/redoc" if serve_frontend else "/redoc",
    )

    # Pydantic's validation detail echoes the offending input. For NaN/Inf
    # that made the 422 response itself unserialisable, so a bad request
    # became a 500.
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": _json_safe(exc.errors())},
        )

    # CORS for frontend dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(APITokenMiddleware)
    # Prometheus request counters/latency. The Grafana dashboard queried
    # api_requests_total/api_duration_seconds, but nothing ever recorded them.
    from quant_platform.utils.metrics import install_http_middleware

    install_http_middleware(app)

    # API routes
    app.include_router(api_router)
    app.include_router(monitor_router)

    # Serve frontend static files in production. Warn loudly when the bundle
    # is older than the source: serving a stale dist silently shipped removed
    # client-side fabrications back to users.
    if serve_frontend and _DIST_DIR.exists():
        index = _DIST_DIR / "index.html"
        source_dir = _FRONTEND_DIR / "src"
        if index.exists() and source_dir.exists():
            newest_source = max(
                (path.stat().st_mtime for path in source_dir.rglob("*") if path.is_file()),
                default=0.0,
            )
            if newest_source > index.stat().st_mtime:
                logging.getLogger(__name__).warning(
                    "frontend/dist is older than frontend/src -- serving a "
                    "stale bundle. Run `npm run build` in frontend/.",
                )
        app.mount("/", StaticFiles(directory=str(_DIST_DIR), html=True), name="frontend")

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="Quant Platform Web Server")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--no-frontend", action="store_true", help="API only, no static files")

    args = parser.parse_args()

    import uvicorn

    serve_frontend = not args.no_frontend
    global app
    app = create_app(serve_frontend=serve_frontend)

    mode = "API + Frontend" if serve_frontend else "API only"
    print("Quant Platform Web Server starting...")
    print(f"  Mode: {mode}")
    print(f"  API:  http://{args.host}:{args.port}/api/docs")
    if serve_frontend:
        if _DIST_DIR.exists():
            print(f"  UI:   http://{args.host}:{args.port}/")
        else:
            print("  UI:   Frontend not built. Run: cd frontend && npm run build")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
