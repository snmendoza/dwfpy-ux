import sys
import numpy
from ctypes import *
import dwfconstants as DConsts
from DigiScopeUI import CenteredViewBox, DigiScopeUI
import math
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import threading
from IPython.display import display, clear_output
import pandas as pd
import numpy as np

import datetime

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")




class DigiScope:
    """
    DigiScope provides an interface to Digilent oscilloscope hardware.
    
    This class handles device connection, configuration, and data acquisition
    from Digilent oscilloscope devices using the Waveforms SDK. It supports
    various trigger modes, channel configurations, and both single and continuous
    acquisition modes.
    
    The class also provides a graphical user interface for real-time visualization
    of acquired waveforms.
    """

    def __init__(self):
        """
        Initialize the DigiScope object and connect to the device.
        
        Sets up the hardware connection, initializes default parameters for
        channels, scope settings, and trigger configuration. Creates the UI
        and prepares the device for data acquisition.
        """
        self.dwf = dwf
        self.DConsts = DConsts
        self.hdwf = c_int()
        self.sts = c_byte()
        self.rgdSamples = (c_double*8192)()
        self.open()
        self.NBuffers = 0

        #params, channel-specific and global
        self.params = {
            1: {    
                "range": 5.0,
                "offset": 0.0,
                "enable": 1,
                "coupling": DConsts.DwfAnalogCouplingAC,
            },
            2: {
                "range": 5.0,
                "offset": 0.0,
                "enable": 1,
                "coupling": DConsts.DwfAnalogCouplingDC,
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
        self.configure_all(self.params)
        self.is_acquiring = False
        self.latest_data = None
        self.data_ready = False
        self.capture_index = 0
        self.capture_time = 0.0
    
        # Initialize last_data with empty arrays for time, ch1, and ch2
        import pandas as pd
        import numpy as np
        self.last_data = pd.DataFrame({
            'time': np.array([]),
            'ch1': np.array([]),
            'ch2': np.array([])
        })
        
        # Create UI only if we're in an environment that supports it
        self.ui = None
        try:
            # Check if we're in a GUI-capable environment
            if QtWidgets.QApplication.instance() is not None:
                self.ui = DigiScopeUI(self, self.params)
        except Exception as e:
            print(f"UI initialization skipped: {e}")

    def open(self):
        """
        Open a connection to the Digilent device.
        
        Attempts to connect to the first available Digilent device with multiple
        retry attempts if the device is busy. Will quit the application if
        connection fails after maximum attempts.
        
        Returns:
            None
        """
        # Try to open the device with multiple attempts
        max_attempts = 5
        attempt = 0
        while attempt < max_attempts:
            dwf.FDwfDeviceOpen(c_int(-1), byref(self.hdwf))
            if self.hdwf.value != DConsts.hdwfNone.value:
                # Successfully opened
                return
            
            # Failed to open, check if it's because device is busy
            szError = create_string_buffer(512)
            dwf.FDwfGetLastErrorMsg(szError)
            error_msg = str(szError.value)
            print(f"Attempt {attempt+1}/{max_attempts}: {error_msg}")
            
            if "Devices are busy" in error_msg:
                # Wait before trying again
                print(f"Device busy, waiting 2 seconds before retry...")
                time.sleep(2)
                attempt += 1
            else:
                # Different error, don't retry
                print("Failed to open device with error: " + error_msg)
                raise Exception(error_msg)
        
        # If we get here, we've exhausted all attempts
        print("Failed to open device after multiple attempts. Please close any other applications using the device.")
        quit()

    def close(self):
        """
        Close the connection to the device and shut down the UI.
        
        Properly terminates the connection to the hardware and closes
        the graphical user interface.
        
        Returns:
            None
        """
        """Close the device"""
        dwf.FDwfDeviceClose(self.hdwf)
        # Close the UI if it exists
        if self.ui is not None:
            try:
                self.ui.close()
            except:
                pass

    def configure_all(self, params):
        """
        Configure all channels, scope settings, and trigger parameters.
        
        Updates the internal parameter dictionary with new values and applies
        them to the device. Configuration is applied in the order: scope settings,
        channel settings, and finally trigger settings.
        
        Args:
            params (dict): Dictionary containing configuration parameters for
                           channels, scope, and trigger settings.
        
        Returns:
            None
        """
        # Update parameters
        if "scope" in params:
            self.params["scope"].update(params["scope"])

        for channel in [1, 2]:
            if channel in params:
                self.params[channel].update(params[channel])

        if "trigger" in params:
            self.params["trigger"].update(params["trigger"])

        # Configuration
        self.dwf.FDwfDeviceAutoConfigureSet(self.hdwf, c_int(0)) # 


        if "wavegen" in params:
            self.generate_waveform(params["wavegen"]["waveform"], \
                                   params["wavegen"]["frequency"], params["wavegen"]["offset"], params["wavegen"]["amplitude"])
        
        # Configure scope first
        self.configure_scope(self.params["scope"])

        # Then configure each channel
        for channel in [1, 2]:
            if channel in self.params:
                self.configure_channel(channel, self.params[channel])

        # Configure trigger
        self.configure_trigger(self.params["trigger"])

        self.dwf.FDwfAnalogInConfigure(self.hdwf, c_int(1), c_int(0))

    def configure_channel(self, channel, params):
        """
        Configure a specific oscilloscope channel.
        
        Sets the enable state, range, offset, and coupling mode for the specified channel.
        
        Args:
            channel (int): Channel number (1 or 2)
            params (dict): Dictionary containing channel parameters:
                           - "enable": 0 (disabled) or 1 (enabled)
                           - "range": Vertical range in volts
                           - "offset": Vertical offset in volts
                           - "coupling": Coupling mode (AC or DC)
        
        Returns:
            None
        """
        # Enable the channel first
        dwf.FDwfAnalogInChannelEnableSet(self.hdwf, c_int(channel-1), c_int(params["enable"]))

        # Set range and offset
        dwf.FDwfAnalogInChannelRangeSet(self.hdwf, c_int(channel-1), c_double(params["range"]))
        dwf.FDwfAnalogInChannelOffsetSet(self.hdwf, c_int(channel-1), c_double(params["offset"]))

        # Set coupling
        dwf.FDwfAnalogInChannelCouplingSet(self.hdwf, c_int(channel-1), params["coupling"])


                
    def configure_scope(self, params):
        """
        Configure global oscilloscope settings.
        
        Sets the sampling frequency and buffer size for data acquisition.
        
        Args:
            params (dict): Dictionary containing scope parameters:
                           - "frequency": Sampling frequency in Hz
                           - "samples": Number of samples to acquire
        
        Returns:
            None
        """
        # get info
        #bsize_min = c_int()
        #bsize_max = c_int()
        #dwf.FDwfAnalogInBufferSizeInfo(self.hdwf, byref(bsize_min), byref(bsize_max)) #buffer min max size
        #nbuffers = int(np.round(bsize_max.value / params["samples"]))

        # Set frequency and buffer size
        dwf.FDwfAnalogInFrequencySet(self.hdwf, c_double(params["frequency"])) #set frequency
        dwf.FDwfAnalogInBufferSizeSet(self.hdwf, c_int(params["samples"])) #set buffer size
        #dwf.FDwfAnalogInBuffersSet(self.hdwf, c_int(nbuffers)) #set num buffers
        #self.NBuffers = c_int(nbuffers)
        # get num buffers
        #dwf.FDwfAnalogInBuffersGet(self.hdwf, byref(self.NBuffers))

        # Set acquisition mode to single
        #dwf.FDwfAnalogInAcquisitionModeSet(self.hdwf, DConsts.acqmodeSingle)

        # Configure the device
        dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(0))

    def configure_trigger(self, params):
        """
        Configure the oscilloscope trigger system.
        
        Sets up the trigger based on the specified type (edge, pulse, or auto).
        
        Args:
            params (dict): Dictionary containing trigger parameters:
                           - "type": Trigger type ("edge", "pulse", or "auto")
                           - "channel": Trigger source channel
                           - "level": Trigger level in volts
                           - "polarity": Trigger slope ("+", "-", or "=")
                           - "position": Trigger position (0-1)
                           Additional parameters for pulse trigger:
                           - "width": Pulse width in seconds
                           - "condition": Comparison condition ("<", ">", or "=")
        
        Returns:
            None
        """
        if params["type"] == "edge":
            self.set_edge_trigger(params["channel"], params["level"], \
                                  params["polarity"], params["position"])
        elif params["type"] == "pulse":
            self.set_pulse_trigger(params["channel"], params["level"], \
                                   params["polarity"], params["width"], \
                                   params["condition"], params["position"])
        elif params["type"] == "auto":
            # Set auto trigger with a reasonable timeout
            dwf.FDwfAnalogInTriggerAutoTimeoutSet(self.hdwf, c_double(1.0))  # 1 second timeout
            dwf.FDwfAnalogInTriggerSourceSet(self.hdwf, DConsts.trigsrcNone)
            # Force configuration to take effect


    def set_edge_trigger(self, channel=1, level=2.0, polarity="+", position=.05,hysteresis=0.001):
        """
        Configure an edge trigger on a specific channel.
        
        Sets up the oscilloscope to trigger on a rising or falling edge.
        
        Args:
            channel (int): Channel to use as trigger source (1 or 2)
            level (float): Voltage level at which to trigger
            polarity (str): Edge direction: "+" for rising, "-" for falling, "=" for either
            position (float): Trigger position in the acquisition window (0-1)
            hysteresis (float): Voltage hysteresis to prevent false triggers
        
        Returns:
            None
        """
        # Map polarity symbols to constants
        polar = {"+": DConsts.DwfTriggerSlopeRise, 
                 "-": DConsts.DwfTriggerSlopeFall,
                 "=": DConsts.DwfTriggerSlopeEither}
        
        # Get buffer size for position calculation
        buffer_size = c_int()
        dwf.FDwfAnalogInBufferSizeGet(self.hdwf, byref(buffer_size))
        
        # Get frequency for position calculation
        frequency = c_double()
        dwf.FDwfAnalogInFrequencyGet(self.hdwf, byref(frequency))
        
        # Disable auto trigger
        dwf.FDwfAnalogInTriggerAutoTimeoutSet(self.hdwf, c_double(0))
        
        # Set trigger source to analog in detector
        dwf.FDwfAnalogInTriggerSourceSet(self.hdwf, DConsts.trigsrcDetectorAnalogIn)
        
        # Set trigger type to edge
        dwf.FDwfAnalogInTriggerTypeSet(self.hdwf, DConsts.trigtypeEdge)
        
        # Set trigger channel (0-based index)
        dwf.FDwfAnalogInTriggerChannelSet(self.hdwf, c_int(channel-1))
        
        # Set trigger level
        dwf.FDwfAnalogInTriggerLevelSet(self.hdwf, c_double(level))
        
        # Set trigger condition (rise/fall/either)
        dwf.FDwfAnalogInTriggerConditionSet(self.hdwf, polar[polarity])
        
        # Set trigger hysteresis
        dwf.FDwfAnalogInTriggerHysteresisSet(self.hdwf, c_double(hysteresis))
        
        # Calculate trigger position in seconds
        # Convert from normalized position [0,1] to seconds relative to buffer
        trigger_position_seconds = (0.5-position) * (buffer_size.value / frequency.value)
        
        # Set trigger position
        dwf.FDwfAnalogInTriggerPositionSet(self.hdwf, c_double(trigger_position_seconds))
        

    def print_trigger_settings(self):
        """
        Print all current trigger settings to the console.
        
        Useful for debugging trigger configuration issues. Displays source,
        type, channel, level, condition, position, hysteresis, length settings,
        and auto timeout.
        
        Returns:
            None
        """
        # Get trigger source
        source = c_int()
        dwf.FDwfAnalogInTriggerSourceGet(self.hdwf, byref(source))
        print(f"Trigger source: {source.value}")

        # Get trigger type 
        ttype = c_int()
        dwf.FDwfAnalogInTriggerTypeGet(self.hdwf, byref(ttype))
        print(f"Trigger type: {ttype.value}")

        # Get trigger channel
        channel = c_int()
        dwf.FDwfAnalogInTriggerChannelGet(self.hdwf, byref(channel)) 
        print(f"Trigger channel: {channel.value+1}")

        # Get trigger level
        level = c_double()
        dwf.FDwfAnalogInTriggerLevelGet(self.hdwf, byref(level))
        print(f"Trigger level: {level.value:.3f}V")

        # Get trigger condition
        condition = c_int()
        dwf.FDwfAnalogInTriggerConditionGet(self.hdwf, byref(condition))
        print(f"Trigger condition: {condition.value}")

        # Get trigger position
        position = c_double()
        dwf.FDwfAnalogInTriggerPositionGet(self.hdwf, byref(position))
        print(f"Trigger position: {position.value:.6f}s")

        # Get trigger hysteresis
        hysteresis = c_double()
        dwf.FDwfAnalogInTriggerHysteresisGet(self.hdwf, byref(hysteresis))
        print(f"Trigger hysteresis: {hysteresis.value:.3f}V")

        # Get trigger length settings (for pulse trigger)
        length = c_double()
        dwf.FDwfAnalogInTriggerLengthGet(self.hdwf, byref(length))
        print(f"Trigger length: {length.value:.6f}s")

        length_condition = c_int()
        dwf.FDwfAnalogInTriggerLengthConditionGet(self.hdwf, byref(length_condition))
        print(f"Trigger length condition: {length_condition.value}")

        # Get auto timeout
        timeout = c_double()
        dwf.FDwfAnalogInTriggerAutoTimeoutGet(self.hdwf, byref(timeout))
        print(f"Auto trigger timeout: {timeout.value:.3f}s")

    def set_pulse_trigger(self, channel=1, level=2.0, polarity="+", \
                        width=50e-6, condition=">", position=.01,hysteresis=0.1):
        """
        Configure a pulse trigger on a specific channel.
        
        Sets up the oscilloscope to trigger on pulses of specific width.
        
        Args:
            channel (int): Channel to use as trigger source (1 or 2)
            level (float): Voltage level at which to trigger
            polarity (str): Edge direction: "+" for rising, "-" for falling, "=" for either
            width (float): Pulse width in seconds
            condition (str): Width comparison: "<" for less than, ">" for greater than, "=" for equal
            position (float): Trigger position in the acquisition window (0-1)
            hysteresis (float): Voltage hysteresis to prevent false triggers
        
        Returns:
            None
        """
        # Map polarity and condition symbols to constants
        polar = {"+": DConsts.DwfTriggerSlopeRise, 
                 "-": DConsts.DwfTriggerSlopeFall,
                 "=": DConsts.DwfTriggerSlopeEither}
        cond = {"<": DConsts.triglenLess, 
                ">": DConsts.triglenMore, 
                "=": DConsts.triglenTimeout}
        
        # Get buffer size for position calculation
        buffer_size = c_int()
        dwf.FDwfAnalogInBufferSizeGet(self.hdwf, byref(buffer_size))
        
        # Get frequency for position calculation
        frequency = c_double()
        dwf.FDwfAnalogInFrequencyGet(self.hdwf, byref(frequency))
        
        # Disable auto trigger
        dwf.FDwfAnalogInTriggerAutoTimeoutSet(self.hdwf, c_double(0))
        
        # Set trigger source to analog in detector
        dwf.FDwfAnalogInTriggerSourceSet(self.hdwf, DConsts.trigsrcDetectorAnalogIn)
        
        # Set trigger type to pulse
        dwf.FDwfAnalogInTriggerTypeSet(self.hdwf, DConsts.trigtypePulse)
        
        # Set trigger channel (0-based index)
        dwf.FDwfAnalogInTriggerChannelSet(self.hdwf, c_int(channel-1))
        
        # Set trigger level
        dwf.FDwfAnalogInTriggerLevelSet(self.hdwf, c_double(level))
        
        # Add hysteresis to prevent false triggers
        dwf.FDwfAnalogInTriggerHysteresisSet(self.hdwf, c_double(hysteresis))
        
        # Set trigger condition (rise/fall/either)
        dwf.FDwfAnalogInTriggerConditionSet(self.hdwf, polar[polarity])
        
        # Set pulse width and condition
        dwf.FDwfAnalogInTriggerLengthSet(self.hdwf, c_double(width))
        dwf.FDwfAnalogInTriggerLengthConditionSet(self.hdwf, cond[condition])
        
        # Calculate trigger position in seconds
        # Convert from normalized position [0,1] to seconds relative to buffer
        trigger_position_seconds = (0.5-position) * (buffer_size.value / frequency.value)
        
        # Set trigger position
        dwf.FDwfAnalogInTriggerPositionSet(self.hdwf, c_double(trigger_position_seconds))
        
        # Apply configuration
        dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(0))
    def force_trigger(self):
        """
        Force a trigger event regardless of trigger conditions.
        
        Useful when waiting for a trigger condition that may not occur.
        
        Returns:
            None
        """
        dwf.FDwfAnalogInTriggerForce(self.hdwf)
        
    def generate_waveform(self, waveform, frequency, offset, amplitude):
        """
        Generate a waveform on the device's analog output.
        
        Useful for testing trigger and acquisition functionality.
        
        Args:
            frequency (float): Waveform frequency in Hz
            offset (float): DC offset in volts
            amplitude (float): Peak amplitude in volts
        
        Returns:
            None
        """
        waveform_map = {"sine": DConsts.funcSine,
                        "square": DConsts.funcSquare,
                        "triangle": DConsts.funcTriangle,
                        "ramp": DConsts.funcRampUp}
        # Configure the analog output
        dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, c_int(0), c_int(0), c_int(1))  # Enable channel 1
        dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf, c_int(0), c_int(0), waveform_map[waveform])
        dwf.FDwfAnalogOutNodeFrequencySet(self.hdwf, c_int(0), c_int(0), c_double(frequency))
        dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf, c_int(0), c_int(0), c_double(offset))
        dwf.FDwfAnalogOutNodeAmplitudeSet(self.hdwf, c_int(0), c_int(0), c_double(amplitude))
        dwf.FDwfAnalogOutConfigure(self.hdwf, c_int(0), c_int(1))  # Start the output

    def start_acquisition(self):
        """
        Start continuous data acquisition in a background thread.
        
        Configures the device for continuous acquisition and starts a thread
        that continuously acquires data and updates the latest_data property.
        
        Returns:
            None
        """
        if not self.is_acquiring:
            self.is_acquiring = True
            self.data_ready = False
            # Configure for continuous acquisition
            dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(1))

            # Start acquisition thread
            self.acquisition_thread = threading.Thread(target=self._acquisition_loop)
            self.acquisition_thread.daemon = True
            self.acquisition_thread.start()

    def stop_acquisition(self):
        """
        Stop the continuous data acquisition.
        
        Terminates the acquisition thread and stops the device acquisition.
        
        Returns:
            None
        """
        if self.is_acquiring:
            self.is_acquiring = False
            # Stop acquisition
            dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(0))
            if hasattr(self, 'acquisition_thread'):
                self.acquisition_thread.join(timeout=1.0)

    
    def _acquisition_loop(self):
        """
        Internal method that runs the continuous acquisition loop.
        
        This method runs in a separate thread and continuously acquires data
        from the device, updating the latest_data property with each acquisition.
        
        Returns:
            None
        """
        capture_index = 0
        while self.is_acquiring:
            try:
                start_time = time.time()

                # Configure for continuous acquisition
                dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(1))

                # Wait for acquisition to complete
                while True:
                    dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(self.sts))
                    if self.sts.value == DConsts.DwfStateDone.value:
                        break
                    time.sleep(0.0001)

                # Get data from both channels
                buffer_size = c_int()
                dwf.FDwfAnalogInBufferSizeGet(self.hdwf, byref(buffer_size))
                ch1_data = (c_double*buffer_size.value)()
                ch2_data = (c_double*buffer_size.value)()

                dwf.FDwfAnalogInStatusData(self.hdwf, 0, ch1_data, buffer_size.value)
                dwf.FDwfAnalogInStatusData(self.hdwf, 1, ch2_data, buffer_size.value)

                # Convert to numpy arrays
                ch1_np = numpy.frombuffer(ch1_data, dtype=numpy.float64)
                ch2_np = numpy.frombuffer(ch2_data, dtype=numpy.float64)

                # Create time axis
                frequency = c_double()
                dwf.FDwfAnalogInFrequencyGet(self.hdwf, byref(frequency))
                time_data = numpy.linspace(0, buffer_size.value/frequency.value, buffer_size.value)

                # Create DataFrame
                import pandas as pd
                df = pd.DataFrame({
                    'time': time_data,
                    'ch1': ch1_np,
                    'ch2': ch2_np
                })

                # Update latest data and metadata
                self.latest_data = df
                self.last_data = df  # Also update last_data for UI access
                self.capture_index = capture_index
                self.capture_time = (time.time() - start_time) * 1000  # Convert to ms
                self.data_ready = True
                capture_index += 1

            except Exception as e:
                print(f"Error in acquisition loop: {e}")
                self.is_acquiring = False
                break

    def acquire_single(self):
        """
        Acquire a single triggered capture.
        
        Configures the device for a single acquisition, waits for the acquisition
        to complete, and returns the acquired data.
        
        Returns:
            pandas.DataFrame: DataFrame containing time, ch1, and ch2 data columns
        """
        nSamples = self.params["scope"]["samples"]
        frequency = self.params["scope"]["frequency"]

        # Set up the channels
        channel1 = (c_double*nSamples)()
        channel2 = (c_double*nSamples)()
        self.dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(1))
        # Acquire the stack
        print("Beginning acquisition")
        # Update capture index for UI
        self.capture_index = 1
            
        # Wait for acquisition to complete
        while True:
            status = c_byte()
            if dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(status)) != 1:
                szError = create_string_buffer(512)
                dwf.FDwfGetLastErrorMsg(szError);
                print("failed to open device\n"+str(szError.value))
                raise Exception("failed to open device")
            if status.value == DConsts.DwfStateDone.value :
                break
            
        #compile data
        self.dwf.FDwfAnalogInStatusData(self.hdwf, 0, channel1, nSamples) # get channel 1 data
        self.dwf.FDwfAnalogInStatusData(self.hdwf, 1, channel2, nSamples) # get channel 2 data
        
        #convert to DF
        df = pd.DataFrame({
            'time': numpy.linspace(0, nSamples/frequency, nSamples),
            'ch1': numpy.frombuffer(channel1, dtype=numpy.float64),
            'ch2': numpy.frombuffer(channel2, dtype=numpy.float64)
        })

        # Update last_data for UI access
        self.last_data = df
        self.data_ready = True

        return df

    def start_live_graph(self, *args, **kwargs):
        """
        Start the live graphical display of acquired data.
        
        Launches the UI's live graph functionality to visualize data in real-time.
        
        Args:
            *args: Variable length argument list passed to UI's start_live_graph
            **kwargs: Arbitrary keyword arguments passed to UI's start_live_graph
        
        Returns:
            None
        """
        if self.ui is not None:
            self.ui.start_live_graph(*args, **kwargs)
        else:
            print("UI not available. Running in non-GUI mode.")

    def acquire_series(self, num_captures, chan=[1, 2], callback=None,verbose=0):
        """
        Acquire a stack of captures using external triggering.
        
        Parameters:
        -----------
        num_captures : int
            Number of captures to acquire
        chan : list
            List of channels to acquire (1, 2, or [1, 2])
        callback : function, optional
            Callback function to call after each acquisition
            
        Returns:
        --------
        list
            List of dictionaries containing the acquired data
        """
        nSamples = self.params["scope"]["samples"]
        frequency = self.params["scope"]["frequency"]
        # Initialize the stack
        stack = []
        
        # Set up the channels
        if isinstance(chan, int):
            chan = [chan]
        channel1 = []
        channel2 = []
        for i in range(num_captures):
            channel1.append((c_double*nSamples)())
            channel2.append((c_double*nSamples)())
        self.dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(1))
        # Acquire the stack
        print("Beginning acquisition")
        try:
            for i in range(num_captures):
                # Update capture index for UI
                self.capture_index = i + 1
                if verbose > 0:
                    display("Acquiring: {}/{}".format(i+1,num_captures))
                    clear_output(wait=True)
                # Wait for acquisition to complete
                while True:
                    status = c_byte()
                    if dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(status)) != 1:
                        szError = create_string_buffer(512)
                        dwf.FDwfGetLastErrorMsg(szError);
                        print("failed to open device\n"+str(szError.value))
                        raise Exception("failed to open device")
                    if status.value == DConsts.DwfStateDone.value :
                        break
                
                #compile data
                self.dwf.FDwfAnalogInStatusData(self.hdwf, 0, channel1[i], nSamples) # get channel 1 data
                self.dwf.FDwfAnalogInStatusData(self.hdwf, 1, channel2[i], nSamples) # get channel 2 data
                
                # Update last_data for UI access
                self.last_data = stack
                self.data_ready = True
        except KeyboardInterrupt:
            print("Keyboard Interrupt at Acquisition {}".format(i))
            return None
            
        #convert to DFs
        stack_df = []
        for i in range(num_captures):
            df = pd.DataFrame({
                'time': numpy.linspace(0, nSamples/frequency, nSamples),
                'ch1': numpy.frombuffer(channel1[i], dtype=numpy.float64),
                'ch2': numpy.frombuffer(channel2[i], dtype=numpy.float64)
            })
            stack_df.append(df)

        return stack_df
    
    def reset_single_triggered(self):
        """
        Reset the device to single-trigger acquisition mode.
        
        Configures the device for single acquisition with manual triggering.
        
        Returns:
            str: Status message
        """
        # Configure for single acquisition
        dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(0))  # Reset
        # Set trigger timeout to 0 (disable auto trigger)
        dwf.FDwfAnalogInTriggerAutoTimeoutSet(self.hdwf, c_double(0))
        return "Single trigger mode set"
    
    def trig_wait(self):
        """
        Wait for the trigger to become active.
        
        Blocks until the device enters the triggered or armed state,
        or until timeout (200ms).
        
        Returns:
            None
        """
        start_time = time.time()
        while True:
            dwf.FDwfAnalogInStatus(self.hdwf, c_int(0), byref(self.sts))
            if self.sts.value == DConsts.DwfStateTriggered.value or \
               self.sts.value == DConsts.DwfStateArmed.value:
                return
            if time.time() - start_time > 0.2:  # Timeout after 200ms
                return
            time.sleep(0.01)
    
    def waveform_wait(self):
        """
        Wait for a waveform to be available.
        
        Blocks until the device indicates that acquisition is complete,
        or until timeout (200ms).
        
        Returns:
            None
        """
        start_time = time.time()
        while True:
            dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(self.sts))
            if self.sts.value == DConsts.DwfStateDone.value:
                return
            if time.time() - start_time > 0.2:  # Timeout after 200ms
                return
            time.sleep(0.01)
    
    def wait(self, timeout=0.1):
        """
        Short wait for device stability.
        
        Simple delay to allow the device to stabilize after configuration changes.
        
        Args:
            timeout (float): Wait time in seconds
        
        Returns:
            None
        """
        time.sleep(timeout)
    
 
    def import_current_data(self, chan=(1), l=None):
        """
        Import current data from specified channels.
        
        Retrieves the most recent data from the device for the specified channels.
        
        Args:
            chan (list or int): Channel(s) to import data from
            l (int): Optional limit on the number of data points to import
        
        Returns:
            dict: Dictionary containing time and channel data arrays
        """
        # Get buffer size
        buffer_size = c_int()
        dwf.FDwfAnalogInBufferSizeGet(self.hdwf, byref(buffer_size))
        
        # Limit buffer size if l is specified
        if l is not None and l < buffer_size.value:
            actual_size = l
        else:
            actual_size = buffer_size.value
        
        # Create a dict to store channel data
        data = {}
        
        # Add time data
        frequency = c_double()
        dwf.FDwfAnalogInFrequencyGet(self.hdwf, byref(frequency))
        data['time'] = numpy.linspace(0, actual_size/frequency.value, actual_size)
        
        # Get data for each requested channel
        for ch in (chan if isinstance(chan, (list, tuple)) else [chan]):
            if ch < 1 or ch > 2:  # Limit to available channels
                continue
            
            ch_data = (c_double*buffer_size.value)()
            dwf.FDwfAnalogInStatusData(self.hdwf, ch-1, ch_data, buffer_size.value)
            
            # Convert to numpy array and limit size if needed
            ch_np = numpy.frombuffer(ch_data, dtype=numpy.float64)[:actual_size]
            data[f'ch{ch}'] = ch_np
        
        # Also store as last_data in DataFrame format for UI access
        import pandas as pd
        df = pd.DataFrame({
            'time': data['time'],
            'ch1': data.get('ch1', numpy.zeros_like(data['time'])),
            'ch2': data.get('ch2', numpy.zeros_like(data['time']))
        })
        self.last_data = df
        
        return data
        
    def set_rolling_acq(self):
        """
        Set the device to continuous (rolling) acquisition mode.
        
        Starts continuous data acquisition in the background.
        
        Returns:
            str: Status message
        """
        self.start_acquisition()
        return "Continuous acquisition started"

def self_trigger_test(params):
    """
    Test the DigiScope with a simple waveform and pulse trigger.
    
    Creates a DigiScope instance, configures it with the provided parameters,
    generates a test waveform, and starts the live graph display.
    
    Args:
        params (dict): Configuration parameters for the scope
    
    Returns:
        None
    """
    params["trigger"] = {"type": "edge", "channel": 1,
                         "level": 1.0, "polarity": "+", 
                         "position": 0.01}
    ds = DigiScope()
    ds.configure_all(params)
    ds.print_trigger_settings()
    ds.generate_waveform("sine",20, 0.0, 1.5)
    #ds.start_live_graph()
    t0 = time.time()
    dfs = ds.acquire_series(1)
    t1 = time.time()
    print(len(dfs))
    print(len(dfs[0]['time']))
    print(f"Time taken: {t1-t0} seconds")


def edge_trigger_test(params):
    """
    Test the DigiScope with edge triggering and external waveform.
    
    Creates a DigiScope instance, configures it with the provided parameters,
    and performs multiple acquisitions with edge triggering.
    
    Args:
        params (dict): Configuration parameters for the scope
    
    Returns:
        None
    """
    ds = DigiScope()
    ds.configure_all(params)
    ds.external_stack_acquire(10, trigger_type="edge")

if __name__ == "__main__":
    #test with GUI
    frequency = 1e6
    duration = 10
    samples = int(frequency * duration)
    custom_params = {
        1: {
            "range": 5.0,
            "offset": 0.0,
            "enable": 1,
            "coupling": DConsts.DwfAnalogCouplingDC,
        },
        2: {
            "range": 5.0,
            "offset": 0.0,
            "enable": 1,
            "coupling": DConsts.DwfAnalogCouplingDC,
        },
        "scope": {
            "frequency": frequency,
            "samples": samples,
        },
        "trigger": {
            "type": "edge",
            "channel": 1,
            "level": 1.0,
            "polarity": "+"
            #"width": 0.05,
            #"condition": ">",
            #"position": 0.5,
        }
    }
    self_trigger_test(custom_params)
    #stack_acquire_test(custom_params)