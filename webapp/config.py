"""
webapp/config.py
────────────────
Centralised settings for the OMR FastAPI service.

All values can be overridden via environment variables (or a .env file
placed in the project root).  Variable names are prefixed with OMR_.

Usage:
    from webapp.config import get_settings
    s = get_settings()
    s.runs_dir          # pathlib.Path
    s.max_upload_bytes  # int (derived from OMR_MAX_UPLOAD_MB)
"""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _PYDANTIC_V2 = True
except ImportError:
    # Fallback: pydantic-settings is installed with pydantic v1 style
    from pydantic import BaseSettings  # type: ignore[no-redef]
    _PYDANTIC_V2 = False


class Settings(BaseSettings):
    # ── Auth ──────────────────────────────────────────────────────────────────
    # Leave unset (empty string) for dev-mode (open, but warns in logs).
    api_key: str = ""

    # ── Storage ───────────────────────────────────────────────────────────────
    runs_dir: pathlib.Path = pathlib.Path("web_runs")

    # ── Upload limits ─────────────────────────────────────────────────────────
    max_upload_mb: int = 25

    # ── Rate limiting (per IP, per minute) ────────────────────────────────────
    rate_limit_per_minute: int = 30

    # ── Concurrency ───────────────────────────────────────────────────────────
    # Max simultaneous OMR pipeline calls per worker process.
    max_concurrent_jobs: int = 2
    # Seconds a request waits for a free job slot before receiving 503.
    queue_wait_s: float = 10.0

    # ── Run lifecycle ─────────────────────────────────────────────────────────
    # Sweeper removes run directories older than this many hours.
    run_ttl_hours: float = 24.0
    # How often the sweeper checks (seconds).
    sweeper_interval_s: float = 300.0

    # ── Debug artifacts ───────────────────────────────────────────────────────
    # Write intermediate debug JPEGs (00_original, 03_warped, 06_grid, etc.)
    # to the run directory so the result page can display the processing steps.
    # Set to False in production to save ~4 MB/run (only result_v3.json kept).
    debug_artifacts: bool = True

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    if _PYDANTIC_V2:
        model_config = SettingsConfigDict(
            env_prefix="OMR_",
            env_file=".env",
            env_file_encoding="utf-8",
        )
    else:
        class Config:
            env_prefix = "OMR_"
            env_file = ".env"
            env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
