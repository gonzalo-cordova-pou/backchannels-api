from unittest.mock import Mock, patch

import pytest

from app.core.logging import RequestLogger


class TestRequestLogger:
    """Unit tests for RequestLogger context manager"""

    @pytest.mark.unit
    def test_request_logger_with_exception(self):
        """Test RequestLogger with exception handling"""
        with patch("app.core.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            try:
                with RequestLogger("test-id", "POST /predict", "test-model"):
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Verify that error was logged
            mock_logger.error.assert_called()


class TestLoggingFunctions:
    """Unit tests for logging utility functions"""

    @pytest.mark.unit
    def test_log_prediction(self):
        """Test log_prediction function"""
        with patch("app.core.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            from app.core.logging import log_prediction

            log_prediction(
                logger=mock_logger,
                text="test utterance",
                prediction=True,
                confidence=0.85,
                model_name="test-model",
                latency_ms=25.0,
                request_id="test-id",
            )

            mock_logger.info.assert_called()

    @pytest.mark.unit
    def test_log_performance_warning(self):
        """Test log_performance_warning function"""
        with patch("app.core.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            from app.core.logging import log_performance_warning

            # Test with performance issue
            log_performance_warning(
                logger=mock_logger,
                operation="test_operation",
                latency_ms=100.0,
                threshold_ms=50.0,
            )

            # Should log a warning for slow performance
            mock_logger.warning.assert_called()

    @pytest.mark.unit
    def test_log_batch_metrics(self):
        """Test log_batch_metrics function"""
        with patch("app.core.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            from app.core.logging import log_batch_metrics

            log_batch_metrics(
                logger=mock_logger,
                batch_size=10,
                total_latency_ms=250.0,
                model_name="test-model",
                request_id="test-id",
            )

            mock_logger.info.assert_called()
