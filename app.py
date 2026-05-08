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
import sys
from pathlib import Path

# Ensure the parent directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from quant_platform.api.routes import router as api_router

# Determine frontend dist path
_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_DIST_DIR = _FRONTEND_DIR / "dist"


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

    # API routes
    app.include_router(api_router)

    # Serve frontend static files in production
    if serve_frontend and _DIST_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_DIST_DIR), html=True), name="frontend")

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="Quant Platform Web Server")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--no-frontend", action="store_true", help="API only, no static files")

    args = parser.parse_args()

    import uvicorn

    serve_frontend = not args.no_frontend
    global app
    app = create_app(serve_frontend=serve_frontend)

    mode = "API + Frontend" if serve_frontend else "API only"
    print(f"Quant Platform Web Server starting...")
    print(f"  Mode: {mode}")
    print(f"  API:  http://{args.host}:{args.port}/api/docs")
    if serve_frontend:
        if _DIST_DIR.exists():
            print(f"  UI:   http://{args.host}:{args.port}/")
        else:
            print(f"  UI:   Frontend not built. Run: cd frontend && npm run build")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
