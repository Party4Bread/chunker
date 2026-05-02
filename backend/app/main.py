"""FastAPI entrypoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .routers import export, infer, ingest, projects, records

# Path to the built SPA (frontend/build/client/). Optional — only mounted if it exists.
FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "build" / "client"


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Chunker Postprocessing API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    app.include_router(projects.router)
    app.include_router(records.router)
    app.include_router(ingest.router)
    app.include_router(infer.router)
    app.include_router(export.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "vllm_base_url": settings.vllm_base_url, "vllm_model": settings.vllm_model}

    # Serve the built SPA at / (and SPA-fallback all unknown non-/api routes to index.html
    # so React Router can handle them client-side).
    if FRONTEND_DIST.exists():
        index_path = FRONTEND_DIST / "index.html"
        # Static assets (Vite emits hashed files under /assets/...)
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str):
            # Anything under /api was already handled by the routers above.
            # For everything else, serve the SPA shell so React Router takes over.
            candidate = FRONTEND_DIST / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(index_path)

    return app


app = create_app()
