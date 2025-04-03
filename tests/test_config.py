"""
Test file for the QTrust configuration module.
"""

import os
import json
import sys
import tempfile
import unittest
import yaml
from pathlib import Path

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrust.utils.config import QTrustConfig, update_config_from_args, parse_arguments

class TestQTrustConfig(unittest.TestCase):
    """Test cases for QTrustConfig class in qtrust/utils/config.py"""
    
    def setUp(self):
        """Set up test environment before each test method."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_file_yaml = os.path.join(self.temp_dir.name, "test_config.yaml")
        self.config_file_json = os.path.join(self.temp_dir.name, "test_config.json")
        
        # Create a test configuration
        self.test_config = {
            "environment": {
                "num_shards": 8,
                "num_nodes_per_shard": 15
            },
            "dqn_agent": {
                "learning_rate": 0.002,
                "gamma": 0.95
            },
            "new_section": {
                "test_param": "test_value"
            }
        }
        
    def tearDown(self):
        """Clean up resources after each test method."""
        self.temp_dir.cleanup()
        
    def test_default_initialization(self):
        """Test initialization with default configuration."""
        config = QTrustConfig()
        
        # Check that the default values are correctly loaded
        self.assertEqual(config.get("environment.num_shards"), 4)
        self.assertEqual(config.get("dqn_agent.learning_rate"), 0.001)
        self.assertEqual(config.get("visualization.save_plots"), True)
        
        # Check a non-existent key
        self.assertIsNone(config.get("non_existent_key"))
        self.assertEqual(config.get("non_existent_key", "default"), "default")
        
    def test_load_yaml_config(self):
        """Test loading configuration from YAML file."""
        # Create a YAML configuration file
        with open(self.config_file_yaml, 'w') as f:
            yaml.dump(self.test_config, f)
            
        config = QTrustConfig(self.config_file_yaml)
        
        # Check that values from the file are loaded
        self.assertEqual(config.get("environment.num_shards"), 8)
        self.assertEqual(config.get("dqn_agent.learning_rate"), 0.002)
        self.assertEqual(config.get("new_section.test_param"), "test_value")
        
        # Check that values not in file remain with default values
        self.assertEqual(config.get("environment.max_steps"), 1000)
        self.assertEqual(config.get("dqn_agent.epsilon_start"), 1.0)
        
    def test_load_json_config(self):
        """Test loading configuration from JSON file."""
        # Create a JSON configuration file
        with open(self.config_file_json, 'w') as f:
            json.dump(self.test_config, f)
            
        config = QTrustConfig(self.config_file_json)
        
        # Check that values from the file are loaded
        self.assertEqual(config.get("environment.num_shards"), 8)
        self.assertEqual(config.get("dqn_agent.learning_rate"), 0.002)
        self.assertEqual(config.get("new_section.test_param"), "test_value")
        
    def test_set_and_get(self):
        """Test setting and getting configuration values."""
        config = QTrustConfig()
        
        # Set new values
        config.set("environment.num_shards", 16)
        config.set("dqn_agent.new_param", "new_value")
        config.set("custom_section.param1", 100)
        
        # Check that the values are correctly set
        self.assertEqual(config.get("environment.num_shards"), 16)
        self.assertEqual(config.get("dqn_agent.new_param"), "new_value")
        self.assertEqual(config.get("custom_section.param1"), 100)
        
    def test_save_config(self):
        """Test saving configuration to file."""
        config = QTrustConfig()
        
        # Modify some values
        config.set("environment.num_shards", 16)
        config.set("new_section.param1", "test_value")
        
        # Save to YAML file
        config.save_config(self.config_file_yaml)
        
        # Load the saved file and check values
        new_config = QTrustConfig(self.config_file_yaml)
        self.assertEqual(new_config.get("environment.num_shards"), 16)
        self.assertEqual(new_config.get("new_section.param1"), "test_value")
        
        # Save to JSON file
        config.save_config(self.config_file_json)
        
        # Load the saved file and check values
        new_config = QTrustConfig(self.config_file_json)
        self.assertEqual(new_config.get("environment.num_shards"), 16)
        self.assertEqual(new_config.get("new_section.param1"), "test_value")
        
    def test_get_all(self):
        """Test getting the entire configuration."""
        config = QTrustConfig()
        all_config = config.get_all()
        
        # Check that all_config is a dictionary with the expected sections
        self.assertIsInstance(all_config, dict)
        self.assertIn("environment", all_config)
        self.assertIn("dqn_agent", all_config)
        self.assertIn("consensus", all_config)
        self.assertIn("routing", all_config)
        self.assertIn("trust", all_config)
        
    def test_get_section(self):
        """Test getting a specific section of the configuration."""
        config = QTrustConfig()
        env_section = config.get_section("environment")
        
        # Check that the section is a dictionary with the expected keys
        self.assertIsInstance(env_section, dict)
        self.assertIn("num_shards", env_section)
        self.assertIn("num_nodes_per_shard", env_section)
        self.assertIn("max_transactions_per_step", env_section)
        
        # Check that a non-existent section returns an empty dictionary
        empty_section = config.get_section("non_existent")
        self.assertEqual(empty_section, {})
        
    def test_invalid_file_format(self):
        """Test loading from an unsupported file format."""
        invalid_file = os.path.join(self.temp_dir.name, "test_config.txt")
        with open(invalid_file, 'w') as f:
            f.write("This is not a valid config file")
            
        # Create a fresh config first to ensure we don't have lingering test config
        original_config = QTrustConfig()
        # Then try to load from the invalid file format
        invalid_config = QTrustConfig(invalid_file)
        
        # Initialization should use default config when file format is invalid
        self.assertEqual(invalid_config.get("environment.num_shards"), 
                         original_config.get("environment.num_shards"))
        
    def test_non_existent_file(self):
        """Test loading from a non-existent file."""
        non_existent_file = os.path.join(self.temp_dir.name, "non_existent.yaml")
        
        # Create a fresh config first to ensure we don't have lingering test config
        original_config = QTrustConfig()
        # Then try to load from a non-existent file
        non_existent_config = QTrustConfig(non_existent_file)
        
        # Initialization should use default config when file doesn't exist
        self.assertEqual(non_existent_config.get("environment.num_shards"), 
                         original_config.get("environment.num_shards"))
        
    def test_update_config_from_args(self):
        """Test updating configuration from command line arguments."""
        config = QTrustConfig()
        
        # Create a mock args object with all required attributes
        class MockArgs:
            def __init__(self):
                self.num_shards = 20
                self.learning_rate = 0.005
                self.config_file = None
                self.seed = 456
                self.num_nodes_per_shard = None
                self.num_episodes = None
                self.num_transactions = None
                self.cross_shard_prob = None
                self.output_dir = None
                self.visualize = None
        
        args = MockArgs()
        
        # Update config from args
        update_config_from_args(config, args)
        
        # Check that the values are updated
        self.assertEqual(config.get("environment.num_shards"), 20)
        self.assertEqual(config.get("dqn_agent.learning_rate"), 0.005)
        self.assertEqual(config.get("environment.seed"), 456)
        
    def test_nested_set(self):
        """Test setting nested keys that don't exist."""
        config = QTrustConfig()
        
        # Set a deeply nested key
        config.set("new_section.nested.deeper.deepest", "deep_value")
        
        # Check that all the intermediate dictionaries and the value are created
        self.assertEqual(config.get("new_section.nested.deeper.deepest"), "deep_value")
        self.assertIsInstance(config.get_section("new_section"), dict)
        self.assertIsInstance(config.get("new_section.nested"), dict)
        self.assertIsInstance(config.get("new_section.nested.deeper"), dict)
        
    def test_directory_creation(self):
        """Test that the output directory is created."""
        # Create a config with a custom output directory
        config_data = {
            "visualization": {
                "output_dir": os.path.join(self.temp_dir.name, "custom_output")
            }
        }
        
        with open(self.config_file_yaml, 'w') as f:
            yaml.dump(config_data, f)
            
        config = QTrustConfig(self.config_file_yaml)
        
        # Check that the directory was created
        output_dir_path = Path(config.get("visualization.output_dir"))
        self.assertTrue(output_dir_path.exists())
        self.assertTrue(output_dir_path.is_dir())
        
    def test_malformed_config_file(self):
        """Test loading from a malformed configuration file."""
        # Create a malformed YAML file
        with open(self.config_file_yaml, 'w') as f:
            f.write("environment:\n  num_shards: 'not an integer'\n  key without value\n")
        
        # Create a fresh config first to ensure we don't have lingering test config    
        original_config = QTrustConfig()
        # Then try to load from the malformed file
        malformed_config = QTrustConfig(self.config_file_yaml)
        
        # Initialization should use default config when file is malformed
        self.assertEqual(malformed_config.get("environment.num_shards"), 
                         original_config.get("environment.num_shards"))

if __name__ == "__main__":
    unittest.main() 