"""
webapp/middleware.py
────────────────────
ASGI middlewares:

  MaxBodySizeMiddleware  — rejects uploads larger than a configured byte
                           limit *before* reading the body, using the
                           Content-Length header as an early indicator and
                           counting bytes on the actual stream as a safety
                           net.  Returns 413 with a clear JSON body.

  RequestIdMiddleware    — assigns a unique X-Request-ID to every request
                           (uses the client-supplied value if present, so
                           callers can propagate their own correlation IDs).
                           The value is injected into the response headers
                           and stored in request.state.request_id for use
                           by loggers and error handlers.
"""

from __future__ import annotations

import uuid
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """
    Reject requests whose body would exceed *max_bytes*.

    Two-stage enforcement:
    1. Content-Length header check — immediate 413 before the body is read.
    2. Streaming check — counts bytes as they arrive; returns 413 if the
       declared size was absent or falsified.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        # Stage 1: trust Content-Length when present.
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": "UPLOAD_TOO_LARGE",
                                "message": (
                                    f"Upload exceeds the {self.max_bytes // (1024*1024)} MB limit."
                                ),
                            }
                        },
                    )
            except ValueError:
                pass  # Malformed Content-Length — let the body check catch it.

        # Stage 2: count bytes on the raw receive channel.
        total = 0
        _original_receive = request._receive  # save before patching

        async def receive_limited():
            nonlocal total
            message = await _original_receive()  # call original, not patched
            if message.get("type") == "http.request":
                chunk = message.get("body", b"")
                total += len(chunk)
                if total > self.max_bytes:
                    return {
                        "type": "http.request",
                        "body": b"",
                        "more_body": False,
                        "_omr_size_exceeded": True,
                    }
            return message

        request._receive = receive_limited

        response = await call_next(request)

        if total > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "UPLOAD_TOO_LARGE",
                        "message": (
                            f"Upload exceeds the {self.max_bytes // (1024*1024)} MB limit."
                        ),
                    }
                },
            )
        return response


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Ensure every request has an X-Request-ID header value.

    - If the client sends X-Request-ID we use that value (useful for
      distributed tracing / log correlation from the caller side).
    - Otherwise we generate a fresh UUID4.

    The value is echoed back in the response headers and stored in
    ``request.state.request_id`` for downstream use (error handlers,
    logger formatters, etc.).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
