"""
Centralized logging configuration for the crash prediction project.

Provides consistent logging setup across all modules with:
- File rotation (10MB max, 5 backups)
- Console output
- Optional JSON formatting for structured logs
- Timestamp + level + module format
"""

import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "extra_data"):
            log_obj["data"] = record.extra_data
        return json.dumps(log_obj)


def setup_logging(
    name: str,
    log_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
    json_format: bool = False,
    console: bool = True,
    file_logging: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up logging with consistent configuration.

    Args:
        name: Logger name (typically __name__ or script name)
        log_dir: Directory for log files. Defaults to project_root/logs/
        level: Logging level (default: INFO)
        json_format: Use JSON formatting for structured logs
        console: Enable console output
        file_logging: Enable file output with rotation
        max_bytes: Maximum log file size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Format strings
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if file_logging:
        if log_dir is None:
            # Default to project_root/logs/
            project_root = Path(__file__).parent.parent
            log_dir = project_root / "logs"
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean the name for use in filename
        safe_name = name.replace(".", "_").replace("/", "_").replace("\\", "_")
        log_file = log_dir / f"{safe_name}_{timestamp}.log"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Also create/update a "latest" symlink or file for convenience
        latest_log = log_dir / f"{safe_name}_latest.log"
        try:
            if latest_log.exists():
                latest_log.unlink()
            # On Windows, symlinks may require admin privileges, so we just copy the path
            latest_log.write_text(str(log_file))
        except (OSError, PermissionError):
            pass  # Skip if we can't create the pointer file

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance. If not already configured, returns a basic logger.

    Use setup_logging() first for full configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context (like request_id) to all log messages.
    """

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger_with_context(name: str, **context) -> LoggerAdapter:
    """
    Get a logger adapter with additional context fields.

    Args:
        name: Logger name
        **context: Additional context fields (e.g., request_id="abc123")

    Returns:
        LoggerAdapter with context
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, context)
