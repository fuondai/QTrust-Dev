"""
Logging system module for QTrust.

This module provides a standardized logging system for the QTrust blockchain platform.
It includes customized formatters with color support, UTF-8 handling for Windows,
and configurable logging to both console and rotating files.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
import io
from pathlib import Path
from typing import Optional, Union, Dict, Any
import datetime

# Setup logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Standard log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO


class CustomFormatter(logging.Formatter):
    """
    Custom formatter with color support for console output.
    """
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    detailed_fmt = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

    FORMATS = {
        logging.DEBUG: blue + detailed_fmt + reset,
        logging.INFO: green + log_fmt + reset,
        logging.WARNING: yellow + detailed_fmt + reset,
        logging.ERROR: red + detailed_fmt + reset,
        logging.CRITICAL: bold_red + detailed_fmt + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class UTF8StreamHandler(logging.StreamHandler):
    """
    Custom StreamHandler with UTF-8 encoding support for Windows
    """
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
            
        super().__init__(stream)
        
        # Only try to wrap with TextIOWrapper if we have a buffer
        try:
            if hasattr(stream, 'buffer'):
                self.stream = io.TextIOWrapper(stream.buffer, encoding='utf-8', errors='backslashreplace')
        except (ValueError, AttributeError, io.UnsupportedOperation):
            # If wrapping fails, just keep the original stream
            pass

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            
            # Check if stream is closed before writing
            if hasattr(stream, 'closed') and stream.closed:
                return
                
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str, 
               level: int = DEFAULT_LOG_LEVEL, 
               console: bool = True, 
               file: bool = True,
               log_dir: Optional[Union[str, Path]] = None, 
               log_format: str = DEFAULT_FORMAT,
               max_file_size_mb: int = 10,
               backup_count: int = 5) -> logging.Logger:
    """
    Create and configure a logger.
    
    Args:
        name: Name of the logger
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to log to console
        file: Whether to log to file
        log_dir: Directory to store log files, if None will use LOG_DIR
        log_format: Log format
        max_file_size_mb: Maximum log file size (MB)
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Create log directory if it doesn't exist
    if log_dir is None:
        log_dir = LOG_DIR
    else:
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
    
    # Add console handler
    if console:
        # Use UTF8StreamHandler instead of default StreamHandler
        console_handler = UTF8StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)
    
    # Add file handler
    if file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"{name}_{timestamp}.log"
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


# Create loggers for the entire application
app_logger = get_logger("qtrust")
simulation_logger = get_logger("qtrust.simulation")
consensus_logger = get_logger("qtrust.consensus")
dqn_logger = get_logger("qtrust.dqn")
trust_logger = get_logger("qtrust.trust")
federated_logger = get_logger("qtrust.federated") 