"""
webapp/errors.py
────────────────
Centralised exception → HTTP response mapping.

Design goals:
  • Clients always receive a stable, machine-readable error envelope:
        {"error": {"code": "<STABLE_CODE>", "message": "<human text>"}}
  • Full exception detail (traceback, str(exc)) is NEVER sent to clients.
  • Every error is logged server-side with exc_info=True so we keep the
    full stack trace, together with the request ID from RequestIdMiddleware.

Register these handlers in main.py:
    from webapp.errors import add_exception_handlers
    add_exception_handlers(app)
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


# ── Stable client-facing codes ────────────────────────────────────────────────

# HTTP status -> (code, message) for well-known 4xx responses.
_HTTP_CODE_MAP: dict[int, tuple[str, str]] = {
    400: ("BAD_REQUEST",         "The request could not be understood or was missing required parameters."),
    401: ("UNAUTHORIZED",        "Authentication is required. Provide a valid X-API-Key header."),
    403: ("FORBIDDEN",           "You do not have permission to access this resource."),
    404: ("NOT_FOUND",           "The requested resource was not found."),
    409: ("CONFLICT",            "The request conflicts with the current state of the resource."),
    413: ("UPLOAD_TOO_LARGE",    "The uploaded file exceeds the maximum allowed size."),
    415: ("UNSUPPORTED_MEDIA",   "The uploaded file type is not supported. Use JPEG or PNG."),
    422: ("VALIDATION_ERROR",    "Request validation failed."),
    429: ("RATE_LIMITED",        "Too many requests. Please wait and try again."),
    503: ("SERVICE_BUSY",        "The service is at capacity. Please retry shortly."),
}

_DEFAULT_5XX = ("INTERNAL_ERROR", "An unexpected error occurred. Please try again later.")


def _error_body(status_code: int, override_message: str | None = None) -> dict:
    code, default_msg = _HTTP_CODE_MAP.get(status_code, _DEFAULT_5XX)
    return {"error": {"code": code, "message": override_message or default_msg}}


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


# ── Handlers ──────────────────────────────────────────────────────────────────

async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    rid = _get_request_id(request)
    status = exc.status_code

    # 4xx: log at WARNING with just the code, no traceback needed.
    # 5xx: log at ERROR.
    log_level = logging.WARNING if status < 500 else logging.ERROR
    logger.log(log_level, "HTTP %d on %s %s  request_id=%s",
               status, request.method, request.url.path, rid)

    # Use the exc.detail only to choose a message for well-known codes;
    # for unknown 4xx we use the stable map so we don't leak internal info.
    override = None
    if status in (401, 403, 404, 429, 503):
        # These details are safe (set by us, not from user input).
        if isinstance(exc.detail, str):
            override = exc.detail

    body = _error_body(status, override)
    headers = dict(exc.headers or {})
    headers["X-Request-ID"] = rid
    return JSONResponse(status_code=status, content=body, headers=headers)


async def _unhandled_exception_handler(request: Request, exc: Exception):
    rid = _get_request_id(request)
    logger.error(
        "Unhandled exception on %s %s  request_id=%s",
        request.method, request.url.path, rid,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content=_error_body(500),
        headers={"X-Request-ID": rid},
    )


def add_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on *app*."""
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception,              _unhandled_exception_handler)
