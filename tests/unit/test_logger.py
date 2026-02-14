"""
Unit tests for structured logging.
"""
import json
import logging
import pytest
from pathlib import Path
from io import StringIO
from app.core.logger import JSONFormatter, setup_logging


def test_json_formatter_basic():
    """Test JSONFormatter produces valid JSON output."""
    formatter = JSONFormatter()

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    output = formatter.format(record)

    # Should be valid JSON
    data = json.loads(output)
    assert data["level"] == "INFO"
    assert data["message"] == "Test message"
    assert data["logger"] == "test"
    assert data["line"] == 10
    assert "timestamp" in data


def test_json_formatter_with_extra_fields():
    """Test JSONFormatter includes extra fields from record."""
    formatter = JSONFormatter()

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    # Add extra fields
    record.request_id = "abc123"
    record.image_id = "img456"
    record.duration_ms = 150
    record.num_coins = 3

    output = formatter.format(record)
    data = json.loads(output)

    assert data["request_id"] == "abc123"
    assert data["image_id"] == "img456"
    assert data["duration_ms"] == 150
    assert data["num_coins"] == 3


def test_json_formatter_with_exception():
    """Test JSONFormatter includes exception info."""
    formatter = JSONFormatter()

    try:
        raise ValueError("Test error")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=10,
        msg="Error occurred",
        args=(),
        exc_info=exc_info,
    )

    output = formatter.format(record)
    data = json.loads(output)

    assert data["level"] == "ERROR"
    assert data["message"] == "Error occurred"
    assert "exception" in data
    assert "ValueError: Test error" in data["exception"]


def test_setup_logging_creates_logger():
    """Test setup_logging creates configured logger."""
    logger = setup_logging(log_level="DEBUG")

    assert logger.name == "coin_detection"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) > 0


def test_setup_logging_console_handler():
    """Test setup_logging adds console handler."""
    logger = setup_logging()

    # Should have at least console handler
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    # Handlers should use JSONFormatter
    for handler in logger.handlers:
        assert isinstance(handler.formatter, JSONFormatter)


def test_setup_logging_with_file(tmp_path):
    """Test setup_logging creates file handler when log_file provided."""
    log_file = tmp_path / "test.log"

    logger = setup_logging(log_file=log_file)

    # Should create log file parent directory
    assert log_file.parent.exists()

    # Should have file handler
    from logging.handlers import RotatingFileHandler

    assert any(isinstance(h, RotatingFileHandler) for h in logger.handlers)


def test_setup_logging_file_rotation(tmp_path):
    """Test log file rotation configuration."""
    from logging.handlers import RotatingFileHandler

    log_file = tmp_path / "test.log"
    logger = setup_logging(log_file=log_file)

    # Find the rotating file handler
    file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            file_handler = handler
            break

    assert file_handler is not None
    assert file_handler.maxBytes == 10 * 1024 * 1024  # 10MB
    assert file_handler.backupCount == 5


def test_logger_does_not_propagate():
    """Test that logger doesn't propagate to root logger."""
    logger = setup_logging()

    assert logger.propagate is False


def test_logger_info_level_logs():
    """Test that INFO level logs are captured."""
    logger = setup_logging(log_level="INFO")

    # Should accept INFO level
    assert logger.isEnabledFor(logging.INFO)

    # Should reject DEBUG level
    assert not logger.isEnabledFor(logging.DEBUG)


def test_logger_debug_level_logs():
    """Test that DEBUG level logs are captured when configured."""
    logger = setup_logging(log_level="DEBUG")

    # Should accept both DEBUG and INFO
    assert logger.isEnabledFor(logging.DEBUG)
    assert logger.isEnabledFor(logging.INFO)
