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
import os
import sys
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
    """Require a bearer token for /api routes when QUANT_API_TOKEN is set."""

    async def dispatch(self, request: Request, call_next):
        token = os.environ.get("QUANT_API_TOKEN", "")
        path = request.url.path
        if token and path.startswith("/api") and path != "/api/health":
            auth = request.headers.get("Authorization", "")
            api_key = request.headers.get("X-API-Key", "")
            if api_key != token and auth != f"Bearer {token}":
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing API token"})
        return await call_next(request)


def create_app(serve_frontend: bool = True) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="A-Share Multi-Factor Quant Platform",
        description="Multi-factor quantitative research platform with LLM-enhanced stock selection",
        version="1.0.0",
        docs_url="/api/docs" if serve_frontend else "/docs",
        redoc_url="/api/redoc" if serve_frontend else "/redoc",
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

    # API routes
    app.include_router(api_router)
    app.include_router(monitor_router)

    # Serve frontend static files in production
    if serve_frontend and _DIST_DIR.exists():
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
