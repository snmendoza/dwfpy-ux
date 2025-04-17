import json
import os
from DwfInterface.DigiScopeGraph import SettingTable
from DwfInterface.DigiScope import DigiScope, defaults

def create_settings_json(filename="test_settings.json"):
    """Create a JSON file with default settings"""
    # Use the default settings from DigiScope
    with open(filename, 'w') as f:
        json.dump(defaults, f, indent=4)
    print(f"Created settings file: {filename}")
    return filename

def test_with_mock_scope():
    """Test SettingTable with a mock scope that doesn't require hardware"""
    
    # Create a mock scope class
    class MockScope:
        def __init__(self):
            # Make a deep copy of defaults to avoid modifying the original
            self.params = json.loads(json.dumps(defaults))
            self.configured_params = None
            
        def configure_all(self, params):
            print("configure_all called with params:")
            # Print a summary of the params for demonstration
            for key in params:
                if isinstance(key, int):
                    print(f"Channel {key}: Range={params[key].get('range')}, Offset={params[key].get('offset')}")
                else:
                    print(f"{key}: {list(params[key].keys())}")
            self.configured_params = params
            
            # Update our internal params with the new values
            for key, value in params.items():
                if key in self.params:
                    self.params[key].update(value)
                else:
                    self.params[key] = value
            
        def save_params(self, path):
            print(f"save_params called with path: {path}")
            with open(path, 'w') as f:
                json.dump(self.params, f, indent=4)
            print(f"Parameters saved to {path}")
    
    # Create the mock scope
    mock_scope = MockScope()
    
    # Create settings file
    settings_file = create_settings_json()
    
    # Create and display the settings table
    table = SettingTable(mock_scope, settings_file)
    table.display()
    
    # Return the components for interaction in the notebook
    return {
        "scope": mock_scope,
        "table": table,
        "settings_file": settings_file
    }

def cleanup_files(filename="test_settings.json"):
    """Clean up the test files"""
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Removed test file: {filename}")

# For use in Jupyter notebook
if __name__ == "__main__":
    print("Run this in a Jupyter notebook using:")
    print(">>> from test_with_json import test_with_mock_scope")
    print(">>> components = test_with_mock_scope()")
    print(">>> display(components['table'].tab)")
    print(">>> display(components['table'].button_box)")
    print("# To clean up after testing:")
    print(">>> from test_with_json import cleanup_files")
    print(">>> cleanup_files()") 