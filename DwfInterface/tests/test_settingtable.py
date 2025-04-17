import os
import json
import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from DwfInterface.DigiScopeGraph import SettingTable

class MockScope:
    def __init__(self, params=None):
        self.params = params or {
            1: {    
                "range": 5.0,
                "offset": 0.0,
                "enable": 1,
                "coupling": "dc",
            },
            2: {
                "range": 5.0,
                "offset": 0.0,
                "enable": 1,
                "coupling": "dc",
            },
            "scope": {
                "frequency": 1e6,
                "samples": 8000000,
            },
            "trigger": {
                "type": "auto",
                "channel": 1,
                "level": 2.0,
                "polarity": "+",
                "position": 0.01,
            },
            "wavegen": {
                "waveform": "sine",
                "frequency": 20,
                "offset": 0.0,
                "amplitude": 1.5
            }
        }
        self.configure_all_called = False
        self.save_params_called = False
        self.saved_path = None
        self.configured_params = None
        
    def configure_all(self, params):
        self.configure_all_called = True
        self.configured_params = params
        
    def save_params(self, path):
        self.save_params_called = True
        self.saved_path = path

class TestSettingTable(unittest.TestCase):
    
    def setUp(self):
        # Create a mock scope
        self.mock_scope = MockScope()
        
        # Create a test JSON file
        self.test_json_path = "test_settings.json"
        with open(self.test_json_path, "w") as f:
            json.dump(self.mock_scope.params, f)
            
    def tearDown(self):
        # Remove the test JSON file
        if os.path.exists(self.test_json_path):
            os.remove(self.test_json_path)
            
    def test_initialization(self):
        # Test initialization with scope
        table = SettingTable(self.mock_scope)
        self.assertEqual(table.scope, self.mock_scope)
        self.assertEqual(table.submit_func, self.mock_scope.configure_all)
        self.assertEqual(table.save_func, self.mock_scope.save_params)
        
        # Test initialization with scope and params path
        table = SettingTable(self.mock_scope, self.test_json_path)
        self.assertEqual(table.params_path, self.test_json_path)
        
    def test_unpile_params(self):
        # Create the table
        table = SettingTable(self.mock_scope)
        
        # Call unpile_params
        table.unpile_params()
        
        # Check that children and tab_titles are populated
        self.assertTrue(len(table.children) > 0)
        self.assertTrue(len(table.tab_titles) > 0)
        self.assertEqual(len(table.children), len(table.tab_titles))
        
    def test_display(self):
        """Test display method without triggering widget validation errors"""
        # Create the table
        with patch('ipywidgets.Tab') as mock_tab, \
             patch('ipywidgets.Button') as mock_button, \
             patch('ipywidgets.HBox') as mock_hbox:
            
            # Set up mocks
            mock_tab_instance = mock_tab.return_value
            mock_save_button = mock_button.return_value
            mock_apply_button = mock_button.return_value
            mock_button_box = mock_hbox.return_value
            
            # Create the table
            table = SettingTable(self.mock_scope)
            
            # Make sure unpile_params doesn't fail
            table.unpile_params = MagicMock()
            
            # Call display
            table.display()
            
            # Check that the tab was created
            mock_tab.assert_called_once()
            
            # Check that buttons were created and callback connected
            self.assertEqual(table.tab, mock_tab_instance)
            self.assertTrue(hasattr(table, 'save_button'))
            self.assertTrue(hasattr(table, 'apply_button'))
            
            # Verify the button callbacks were set
            mock_save_button.on_click.assert_called()
            mock_apply_button.on_click.assert_called()
        
    def test_on_apply_clicked(self):
        # Create the table
        table = SettingTable(self.mock_scope)
        
        # Mock pile_params
        table.pile_params = MagicMock(return_value=self.mock_scope.params)
        
        # Call on_apply_clicked
        table.on_apply_clicked(None)
        
        # Check that configure_all was called
        self.assertTrue(self.mock_scope.configure_all_called)
        self.assertEqual(self.mock_scope.configured_params, self.mock_scope.params)
        
    def test_on_save_clicked(self):
        # Create the table
        table = SettingTable(self.mock_scope, self.test_json_path)
        
        # Mock pile_params
        table.pile_params = MagicMock(return_value=self.mock_scope.params)
        
        # Call on_save_clicked
        table.on_save_clicked(None)
        
        # Check that save_params was called with the correct path
        self.assertTrue(self.mock_scope.save_params_called)
        self.assertEqual(self.mock_scope.saved_path, self.test_json_path)
        
    def test_pile_params(self):
        # Create the table
        table = SettingTable(self.mock_scope)
        
        # Create mock widgets
        mock_range_widget = MagicMock()
        mock_range_widget.value = 10.0
        
        mock_offset_widget = MagicMock()
        mock_offset_widget.value = 1.0
        
        # Set up param_widgets dictionary
        table.param_widgets = {
            1: {
                "range": mock_range_widget,
                "offset": mock_offset_widget
            }
        }
        
        # Call pile_params
        params = table.pile_params()
        
        # Check that params are correctly generated
        self.assertEqual(params[1]["range"], 10.0)
        self.assertEqual(params[1]["offset"], 1.0)
        
if __name__ == "__main__":
    unittest.main() 