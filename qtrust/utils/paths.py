"""
Path utilities for the QTrust blockchain system.
"""
import os
from pathlib import Path
from typing import Optional

# Path to project directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Path to charts directory
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")
CHARTS_TEST_DIR = os.path.join(CHARTS_DIR, "test")
CHARTS_VISUALIZATION_DIR = os.path.join(CHARTS_DIR, "visualization")
CHARTS_BENCHMARK_DIR = os.path.join(CHARTS_DIR, "benchmark")
CHARTS_SIMULATION_DIR = os.path.join(CHARTS_DIR, "simulation")

# Path to documentation directory
DOCS_DIR = os.path.join(PROJECT_ROOT, "documentation")
DOCS_ARCHITECTURE_DIR = os.path.join(DOCS_DIR, "architecture")
DOCS_BENCHMARK_DIR = os.path.join(DOCS_DIR, "benchmark")
DOCS_METHODOLOGY_DIR = os.path.join(DOCS_DIR, "methodology")
DOCS_GUIDES_DIR = os.path.join(DOCS_DIR, "guides")
DOCS_API_DIR = os.path.join(DOCS_DIR, "api")

# Ensure directories exist
def ensure_dir_exists(directory: str) -> None:
    """Ensure the directory exists, create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def get_chart_path(filename: str, category: str = None) -> str:
    """
    Get the full path for a chart file.
    
    Args:
        filename: Chart filename
        category: Chart category (visualization, benchmark, simulation, test, etc.)
        
    Returns:
        str: Full path to the chart file
    """
    # If category is provided, use the corresponding subdirectory
    if category:
        chart_dir = os.path.join(get_project_root(), "charts", category)
    else:
        chart_dir = os.path.join(get_project_root(), "charts")
        
    # Ensure the directory exists
    os.makedirs(chart_dir, exist_ok=True)
    
    return os.path.join(chart_dir, filename)

def get_doc_path(relative_path: str) -> str:
    """
    Get the full path for a document. Automatically creates directories if needed.
    
    Args:
        relative_path: Relative path within the docs directory
        
    Returns:
        str: Absolute path to the document
    """
    root = get_project_root()
    doc_dir = os.path.join(root, "docs")
    
    # Create docs directory if it doesn't exist
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
    
    # Get the full path
    full_path = os.path.join(doc_dir, relative_path)
    
    # Create subdirectory if needed
    dir_path = os.path.dirname(full_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    return full_path

def get_data_path(relative_path: str) -> str:
    """
    Get the full path for data. Automatically creates directories if needed.
    
    Args:
        relative_path: Relative path within the data directory
        
    Returns:
        str: Absolute path to the data
    """
    root = get_project_root()
    data_dir = os.path.join(root, "data")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Get the full path
    full_path = os.path.join(data_dir, relative_path)
    
    # Create subdirectory if needed
    dir_path = os.path.dirname(full_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    return full_path

# Ensure all directories exist
def ensure_all_dirs() -> None:
    """Ensure all necessary directories exist."""
    for directory in [
        CHARTS_DIR, CHARTS_TEST_DIR, CHARTS_VISUALIZATION_DIR, 
        CHARTS_BENCHMARK_DIR, CHARTS_SIMULATION_DIR,
        DOCS_DIR, DOCS_ARCHITECTURE_DIR, DOCS_BENCHMARK_DIR, 
        DOCS_METHODOLOGY_DIR, DOCS_GUIDES_DIR, DOCS_API_DIR
    ]:
        ensure_dir_exists(directory)

# Initialize directories
ensure_all_dirs() 