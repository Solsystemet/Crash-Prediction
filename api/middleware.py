"""
Request middleware for API observability.

Provides:
- Request ID generation and propagation
- Request/response logging with context
- Latency tracking per request
"""

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Context variable for request ID - available throughout request lifecycle
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Get the current request ID from context.
    
    Returns empty string if called outside of request context.
    """
    return request_id_var.get()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware that adds request context for observability.
    
    Features:
    - Generates unique request ID for each request
    - Adds request ID to response headers
    - Logs request start/end with timing
    - Makes request ID available via context variable
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Set context variable for use in downstream code
        token = request_id_var.set(request_id)
        
        # Store on request state for easy access
        request.state.request_id = request_id
        
        # Log request start
        start_time = time.perf_counter()
        logger.info(
            f"[{request_id[:8]}] {request.method} {request.url.path} started",
            extra={"request_id": request_id, "method": request.method, "path": request.url.path}
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Add headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
            
            # Log request completion
            logger.info(
                f"[{request_id[:8]}] {request.method} {request.url.path} completed "
                f"status={response.status_code} latency={latency_ms:.2f}ms",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                }
            )
            
            return response
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request_id[:8]}] {request.method} {request.url.path} failed "
                f"error={type(e).__name__} latency={latency_ms:.2f}ms",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "latency_ms": latency_ms,
                },
                exc_info=True,
            )
            raise
        finally:
            # Reset context variable
            request_id_var.reset(token)


def setup_middleware(app: FastAPI) -> None:
    """Configure all observability middleware for a FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    app.add_middleware(RequestContextMiddleware)
    logger.info("Request context middleware configured")


class RequestContextFilter(logging.Filter):
    """Logging filter that adds request context to log records.
    
    Adds request_id field to all log records, making it available
    in log formatters.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


def setup_request_context_logging() -> None:
    """Add request context filter to root logger.
    
    Call this once at application startup to include request IDs
    in all log output.
    """
    root_logger = logging.getLogger()
    root_logger.addFilter(RequestContextFilter())
