from __future__ import annotations

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def _error_response(code: str, message: str, details=None, status_code: int = 400) -> JSONResponse:
    payload = {"code": code, "message": message, "details": details}
    return JSONResponse(status_code=status_code, content=payload)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):  # noqa: ANN001
        if isinstance(exc.detail, dict) and "code" in exc.detail and "message" in exc.detail:
            payload = exc.detail
        else:
            payload = {
                "code": f"HTTP_{exc.status_code}",
                "message": str(exc.detail) if exc.detail else "HTTP error",
                "details": None,
            }
        logger.warning("HTTPException: %s", payload)
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request, exc: Exception):  # noqa: ANN001
        logger.exception("Unhandled exception: %s", exc)
        return _error_response(
            code="internal_error",
            message="Internal server error",
            details=str(exc),
            status_code=500,
        )

