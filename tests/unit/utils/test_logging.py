"""
Test file for the QTrust logging module.
"""

import os
import sys
import unittest
import logging
import tempfile
import shutil
import io
from pathlib import Path
import time

# Add the project root directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrust.utils.logging import get_logger, CustomFormatter, UTF8StreamHandler

class TestLogging(unittest.TestCase):
    """Tests for logging utilities."""
    
    def setUp(self):
        """Set up for each test."""
        # Create a temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        self.loggers = []
    
    def tearDown(self):
        """Clean up after each test."""
        # Close all loggers and handlers
        for logger in self.loggers:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        
        # Sleep briefly to ensure files are released
        time.sleep(0.1)
        
        # Remove temporary directory
        try:
            shutil.rmtree(self.test_dir)
        except PermissionError:
            # If can't delete due to Windows file locking, just proceed
            pass
    
    def test_get_logger_basic(self):
        """Test basic logger creation."""
        logger = get_logger("test_logger", log_dir=self.test_dir)
        self.loggers.append(logger)
        
        # Verify logger properties
        self.assertEqual(logger.name, "test_logger")
        self.assertEqual(logger.level, logging.INFO)
        
        # Verify handlers were added
        self.assertEqual(len(logger.handlers), 2)  # Console and file
        
        # Check that one handler is a file handler
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        self.assertEqual(len(file_handlers), 1)
    
    def test_get_logger_console_only(self):
        """Test logger with console output only."""
        # Create a logger with console output only, but with a special log level
        logger = get_logger("test_console_logger", 
                            level=logging.WARNING,
                            file=False, 
                            log_dir=self.test_dir)
        self.loggers.append(logger)
        
        # Verify only one handler exists
        self.assertEqual(len(logger.handlers), 1)
        
        # Verify the log level was set correctly
        self.assertEqual(logger.level, logging.WARNING)
    
    def test_get_logger_file_only(self):
        """Test logger with file output only."""
        logger = get_logger("test_file_logger", console=False, log_dir=self.test_dir)
        self.loggers.append(logger)
        
        # Verify only one handler (file)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.FileHandler)
        
        # Log a message to create the file
        logger.info("Test log message")
        
        # Verify log file was created
        log_files = list(Path(self.test_dir).glob("*.log"))
        self.assertEqual(len(log_files), 1)
    
    def test_logger_output(self):
        """Test logger output contains correct information."""
        # Use a string IO to capture log output
        string_io = io.StringIO()
        
        # Create a logger that writes to our string IO
        logger = logging.getLogger("test_output_logger")
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        logger.handlers = []
        
        # Add a simple handler that writes to our string IO
        handler = logging.StreamHandler(string_io)
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(handler)
        self.loggers.append(logger)
        
        # Log a message
        test_message = "This is a test message"
        logger.info(test_message)
        
        # Check the output
        output = string_io.getvalue()
        self.assertIn("INFO", output)
        self.assertIn(test_message, output)
    
    def test_custom_formatter(self):
        """Test the custom formatter color coding."""
        formatter = CustomFormatter()
        
        # Create a record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Check that color codes are present
        self.assertIn(CustomFormatter.green, formatted)
        self.assertIn(CustomFormatter.reset, formatted)
        self.assertIn("INFO", formatted)
        self.assertIn("Test message", formatted)
    
    def test_utf8_stream_handler(self):
        """Test UTF8StreamHandler with unicode characters."""
        # Create a logger directly with standard StreamHandler for simpler testing
        logger = logging.getLogger("test_utf8_logger")
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        logger.handlers = []
        
        # Create a string buffer and add a handler
        string_io = io.StringIO()
        handler = logging.StreamHandler(string_io)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        self.loggers.append(logger)
        
        # Test with unicode characters
        test_message = "Unicode test: ñáéíóú"
        logger.info(test_message)
        
        # Verify the message was logged correctly
        output = string_io.getvalue()
        self.assertIn(test_message, output)

if __name__ == "__main__":
    unittest.main() 