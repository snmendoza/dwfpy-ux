# dwfpy_ux
Digilent Waveforms API facade with optional UIX

While the WaveForms software is quite mature, more direct/programmatic communication with these devices might be needed for some experiments.
This library is intended to provide both a lightweight display and communication interface to the oscilloscope.

![DigiScope GUI Example](docs/output.gif)
```python
from DwfInterface import DigiScope
ds = DigiScope()
ds.configure_all(params)
ds.graph()
ds.acquire_continuous()
```

## Overview
`dwfpy_ux` provides a Python interface to Digilent oscilloscope hardware using the Waveforms SDK. This library simplifies device connection, configuration, and data acquisition for Digilent oscilloscope devices.

## Features
- Easy configuration with a single dictionary passed to `configure_all()`
- Data export as pandas DataFrames
- Real-time data visualization GUI for monitoring oscilloscope signals

## Requirements
- Python 3.6+
- NumPy
- Pandas
- Digilent Waveforms SDK (dwf library)
- PyQt5
- pyqtgraph
- pglive

## Installation
1. Install the Digilent Waveforms SDK from [Digilent's website](https://digilent.com/reference/software/waveforms/waveforms-3/start)
2. Install the package using pip:
```
pip install dwfpy_ux
```

Or install from source:
```
git clone https://github.com/snmendoza/dwfpy_ux.git
cd dwfpy_ux
pip install -e .
```

## Quick Start
```python
from DwfInterface import DigiScope
from DwfInterface import dwfconstants as DConsts

# Create a DigiScope instance
ds = DigiScope()

# Configure oscilloscope parameters
params = {
    1: {    
        "range": 5.0, # ambipolar range relative to offset
        "offset": 0.0, # offset from ground
        "enable": 1, # Boolean
        "coupling": "dc", # "ac", "dc", "gnd"
    },
    2: {
        "range": 5.0,
        "offset": 0.0,
        "enable": 1,
        "coupling": "dc",
    },
    "scope": {
        "frequency": 1e6, #Sampling rate
        "samples": 8000, #Number of samples to acquire
                         # Acquisition time = samples / frequency
    },
    "trigger": {
        "type": "edge", # "edge", "pulse", "auto", other not implemented
        "channel": 1, 
        "level": 2.0, 
        "position": 0.01, 
        "hysteresis": 0.01, 
        # for edge, pass channel, level, position, hysteresis, polarity (+/-)
        # for pulse, pass channel, level, polarity (+/-)
        # width, condition (>,<)
    },
    "wavegen": {
        "waveform": "sine", 
        ### Implemented: 
        #"triangle", "ramp", "dc", "noise", "pulse", "sine", #"square", 

        ### Partially Implemented:
        # "trap", "rampdown", "sinepower", #"sinena"
        "frequency": 1e6, #
        "amplitude": 1.0,
        "offset": 0.0,
    }
}

# Apply configuration
ds.configure_all(params)

# Acquire a single capture
data = ds.acquire_single()

# Data is a pandas DataFrame with time, ch1, ch2 columns
print(data.head())

# Close the device when done
ds.close()
```
Example for repeated acquisitions:
```python
ds = DigiScope()

# Configure oscilloscope parameters
params = {...}

# Apply configuration
ds.configure_all(params)

# Acquire a single capture
data = ds.acquire_series(10, verbose=1)

# Data is a pandas DataFrame with time, ch1, ch2 columns
print(data.head())

# Close the device when done
ds.close()
```

## Using the GUI
The library includes a real-time visualization interface for monitoring oscilloscope data:

![DigiScope GUI Example](docs/ux_ex.jpg)

```python
from DwfInterface import DigiScope
from DwfInterface.DigiScopeGraph import OscilloscopeUI
import sys
from PyQt5.QtWidgets import QApplication

# Initialize Qt application in main thread
app = get_qt_app()

# Create DigiScope instance
ds = DigiScope()
ds.configure_all(my_params)

# Create UI (will now show the main window)
ds.graph()

print("UI should be visible now. Starting data acquisition...")

# Start acquiring data in a background thread
def run_acquisition():
    # Wait briefly for UI to initialize
    time.sleep(0.5)
    # Acquire continuous data
    ds.acquire_continuous()
    
acquisition_thread = threading.Thread(target=run_acquisition)
acquisition_thread.daemon = True
acquisition_thread.start()

# This blocks until the window is closed
print("Running Qt event loop in main thread. Close window to exit.")
sys.exit(app.exec_())
```

The GUI displays real-time waveforms from both channels with the following features:
- Live updating plots for Channel 1 and Channel 2
- Auto-scaling for optimal signal viewing
- Synchronized x-axis between both channels
- Efficient rendering with configurable sample downsampling

## Recommended Usage: Jupyter Notebook
For interactive development and experimentation, I recommend using `dwfpy_ux` within Jupyter notebooks which provides a convenient environment for configuring and controlling your Digilent oscilloscope:

```python
import sys
import os
# Import from package
from DwfInterface import DigiScope
# Import required constants
from DwfInterface.dwfconstants import DwfAnalogCouplingDC

# Create a DigiScope instance
ds = DigiScope()

# Configure oscilloscope parameters
params = {....
}
ds.configure_all(params)

# Launch the UI within Jupyter
ds.jupyter_graph()

# Start continuous acquisition
# (this will show live data in the graph window)
ds.acquire_continuous()

# Alternatively, capture a series of acquisitions
series = ds.acquire_series(20)
```

The `jupyter_graph()` method integrates the Qt event loop with Jupyter's event loop, allowing you to interact with both the notebook and the oscilloscope UI simultaneously.

## License
None
