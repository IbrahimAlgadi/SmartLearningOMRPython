"""
webapp/sweeper.py
─────────────────
Background asyncio task that removes stale run directories from RUNS_ROOT.

A "run directory" is any direct child directory of RUNS_ROOT whose *last
modification time* is older than the configured TTL.  Only directories are
removed; stray files at the root level are left alone.

Usage (called from the lifespan context in main.py):

    task = asyncio.create_task(
        sweeper_loop(settings.runs_dir, settings.run_ttl_hours,
                     interval_s=settings.sweeper_interval_s)
    )
    # … at shutdown:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import shutil
import time

logger = logging.getLogger(__name__)


async def sweeper_loop(
    runs_root: pathlib.Path,
    ttl_hours: float,
    interval_s: float = 300.0,
) -> None:
    """
    Infinite loop: every *interval_s* seconds scan *runs_root* and delete
    any subdirectory whose mtime is older than *ttl_hours* hours.

    Designed to run as a background asyncio task; cancelled cleanly via
    ``task.cancel()`` followed by ``await task``.
    """
    ttl_s = ttl_hours * 3600.0
    logger.info(
        "Sweeper started: runs_root=%s  ttl=%.1fh  interval=%.0fs",
        runs_root, ttl_hours, interval_s,
    )

    while True:
        try:
            await asyncio.sleep(interval_s)
            _sweep_once(runs_root, ttl_s)
        except asyncio.CancelledError:
            logger.info("Sweeper task cancelled — shutting down.")
            raise
        except Exception:
            logger.exception("Sweeper encountered an error (will retry next cycle).")


def _sweep_once(runs_root: pathlib.Path, ttl_s: float) -> None:
    """Synchronous sweep — called from the async loop via await asyncio.sleep."""
    if not runs_root.exists():
        return

    now = time.time()
    deleted = 0
    errors  = 0

    for entry in runs_root.iterdir():
        if not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue

        age_s = now - mtime
        if age_s < ttl_s:
            continue

        try:
            shutil.rmtree(entry, ignore_errors=False)
            logger.debug("Sweeper removed: %s  (age=%.1fh)", entry.name, age_s / 3600)
            deleted += 1
        except Exception:
            logger.warning("Sweeper could not remove %s", entry, exc_info=True)
            errors += 1

    if deleted or errors:
        logger.info("Sweeper cycle: deleted=%d  errors=%d", deleted, errors)
