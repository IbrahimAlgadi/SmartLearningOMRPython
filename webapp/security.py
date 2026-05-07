"""
webapp/security.py
──────────────────
FastAPI dependency that enforces X-API-Key authentication.

In dev mode (OMR_API_KEY is empty / not set) the check is bypassed and a
one-time WARNING is logged to remind the operator to set a key before going
to production.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import HTTPException, Request, status

from webapp.config import get_settings

logger = logging.getLogger(__name__)

_dev_mode_warned = False


async def verify_api_key(request: Request) -> None:
    """
    FastAPI dependency — inject as ``Depends(verify_api_key)``.

    Reads the ``X-API-Key`` header and compares it against
    ``settings.api_key`` using a constant-time comparison to prevent
    timing-attack disclosure.

    Dev mode: if ``OMR_API_KEY`` is unset or empty the check is skipped
    and a single WARNING is emitted per process lifetime.
    """
    global _dev_mode_warned

    settings = get_settings()

    if not settings.api_key:
        if not _dev_mode_warned:
            logger.warning(
                "OMR_API_KEY is not set — API is OPEN to anyone on the "
                "network.  Set OMR_API_KEY=<secret> before production use."
            )
            _dev_mode_warned = True
        return

    provided = request.headers.get("X-API-Key", "")
    # secrets.compare_digest is constant-time to prevent timing attacks.
    if not secrets.compare_digest(provided.encode(), settings.api_key.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
