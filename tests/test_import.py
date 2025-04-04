"""
Test imports for the main modules of the QTrust system.

This module validates that all important components can be imported successfully.
"""

import unittest
import os
import sys
import importlib

# Add the root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestImports(unittest.TestCase):
    """Test importing the main modules of QTrust."""
    
    def test_import_qtrust(self):
        """Test importing the main package."""
        import qtrust
        self.assertTrue(hasattr(qtrust, '__name__'))
    
    def test_import_consensus(self):
        """Test importing the consensus module."""
        try:
            from qtrust import consensus
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import consensus module")
    
    def test_import_agents(self):
        """Test importing the agents module."""
        try:
            from qtrust import agents
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import agents module")
    
    def test_import_utils(self):
        """Test importing the utils module."""
        try:
            from qtrust import utils
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import utils module")
    
    def test_import_paths(self):
        """Test importing the paths module."""
        try:
            from qtrust.utils import paths
            self.assertTrue(hasattr(paths, 'PROJECT_ROOT'))
            self.assertTrue(hasattr(paths, 'CHARTS_DIR'))
            self.assertTrue(hasattr(paths, 'DOCS_DIR'))
        except ImportError:
            self.fail("Failed to import paths module")
    
    def test_import_specific_modules(self):
        """Test importing specific important modules."""
        modules_to_test = [
            'qtrust.utils.visualization',
            'qtrust.consensus.adaptive_consensus',
            'qtrust.consensus.adaptive_pos'
        ]
        
        for module_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, '__name__'))
            except ImportError as e:
                self.fail(f"Failed to import {module_name}: {e}")


if __name__ == "__main__":
    unittest.main() 