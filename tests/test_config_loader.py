import unittest
import json
import os
import sys
from unittest.mock import patch, mock_open

# add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.loader import load_config

class TestConfigLoader(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    @patch("os.path.exists")
    def test_load_config_success(self, mock_exists, mock_file):
        # setup
        mock_exists.return_value = True
        
        # execute
        result = load_config("dummy_config.json")
        
        # verify
        self.assertEqual(result, {"key": "value"})
        mock_file.assert_called_with("dummy_config.json", 'r')

    @patch("os.path.exists")
    def test_load_config_file_not_found(self, mock_exists):
        # setup
        mock_exists.return_value = False
        
        # execute and verify
        # since load_config calls quit(), expect SystemExit
        with self.assertRaises(SystemExit):
            load_config("non_existent.json")

if __name__ == '__main__':
    unittest.main()