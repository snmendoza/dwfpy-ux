import sys
import numpy
from ctypes import *
import dwfconstants as DConsts
from DigiScopeGraph import OscilloscopeUI, get_qt_app
import math
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import threading

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
        self.data_connectors = None

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
        
        # UI functionality temporarily disabled
        self.ui = None
        """
        # Create UI only if we're in an environment that supports it
        self.ui = None
        try:
            # Check if we're in a GUI-capable environment
            if QtWidgets.QApplication.instance() is not None:
                self.ui = DigiScopeUI(self, self.params)
        except Exception as e:
            print(f"UI initialization skipped: {e}")
        """

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
        ## set params for shared keywords
        if "holdoff" in params:
            dwf.FDwfAnalogInTriggerHoldoffSet(self.hdwf, c_double(params["holdoff"]))
        if "hysteresis" in params:
            dwf.FDwfAnalogInTriggerHysteresisSet(self.hdwf, c_double(params["hysteresis"]))
        if "position" in params:
            # Calculate trigger position in seconds
            # Convert from normalized position [0,1] to seconds relative to buffer
            buffer_size = c_int()
            dwf.FDwfAnalogInBufferSizeGet(self.hdwf, byref(buffer_size))
            frequency = c_double()
            dwf.FDwfAnalogInFrequencyGet(self.hdwf, byref(frequency))
            trigger_position_seconds = (0.5-params["position"]) * (buffer_size.value / frequency.value)
            dwf.FDwfAnalogInTriggerPositionSet(self.hdwf, c_double(trigger_position_seconds))

        ## set params for specific trigger types
        if params["type"] == "edge":
            self.set_edge_trigger(params["channel"], params["level"], \
                                  params["polarity"])
        elif params["type"] == "pulse":
            self.set_pulse_trigger(params["channel"], params["level"], \
                                   params["polarity"], params["width"], \
                                   params["condition"])
        elif params["type"] == "auto":
            # Set auto trigger with a reasonable timeout
            dwf.FDwfAnalogInTriggerAutoTimeoutSet(self.hdwf, c_double(1.0))  # 1 second timeout
            dwf.FDwfAnalogInTriggerSourceSet(self.hdwf, DConsts.trigsrcNone)
        # Apply the configuration to make sure all settings take effect
        dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(1))



    def set_edge_trigger(self, channel=1, level=2.0, polarity="+"):
        """
        Configure an edge trigger on a specific channel.
        
        Sets up the oscilloscope to trigger on a rising or falling edge.
        
        Args:
            channel (int): Channel to use as trigger source (1 or 2)
            level (float): Voltage level at which to trigger
            polarity (str): Edge direction: "+" for rising, "-" for falling, "=" for either
        
        Returns:
            None
        """
        # Try using the proper constants for edge triggering with enhanced hysteresis
        if polarity == "+":
            slope = DConsts.DwfTriggerSlopeRise
        elif polarity == "-":
            slope = DConsts.DwfTriggerSlopeFall
        else:
            slope = DConsts.DwfTriggerSlopeEither

        
        # First reset all trigger settings to defaults
        dwf.FDwfAnalogInTriggerChannelSet(self.hdwf, c_int(0))  # Default to first channel
        dwf.FDwfAnalogInTriggerTypeSet(self.hdwf, c_int(0))     # Default to edge trigger
        dwf.FDwfAnalogInTriggerConditionSet(self.hdwf, c_int(0)) # Default to rising edge
        dwf.FDwfAnalogInTriggerLevelSet(self.hdwf, c_double(0)) # Default trigger level
        
        # Disable auto trigger to ensure we only trigger on the specified condition
        dwf.FDwfAnalogInTriggerAutoTimeoutSet(self.hdwf, c_double(0))
        
        # Set trigger source to analog in detector
        dwf.FDwfAnalogInTriggerSourceSet(self.hdwf, DConsts.trigsrcDetectorAnalogIn)
        
        # Make sure we're in edge trigger mode
        dwf.FDwfAnalogInTriggerTypeSet(self.hdwf, DConsts.trigtypeEdge)
        
        # Set the trigger channel (0-based index)
        dwf.FDwfAnalogInTriggerChannelSet(self.hdwf, c_int(channel-1))
        
        # Set the trigger level
        dwf.FDwfAnalogInTriggerLevelSet(self.hdwf, c_double(level))
        
        # Set the trigger slope to rise/fall/either
        dwf.FDwfAnalogInTriggerConditionSet(self.hdwf, slope)
    
        

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
                        width=50e-6, condition=">"):
        """
        Configure a pulse trigger on a specific channel.
        
        Sets up the oscilloscope to trigger on pulses of specific width.
        
        Args:
            channel (int): Channel to use as trigger source (1 or 2)
            level (float): Voltage level at which to trigger
            polarity (str): Edge direction: "+" for rising, "-" for falling, "=" for either
            width (float): Pulse width in seconds
            condition (str): Width comparison: "<" for less than, ">" for greater than, "=" for equal
        
        Returns:
            None
        """
        # Map polarity and condition symbols to constants
        # Use the correct constants for pulse trigger
        polar = {"+": DConsts.DwfTriggerSlopeRise, 
                 "-": DConsts.DwfTriggerSlopeFall,
                 "=": DConsts.DwfTriggerSlopeEither}
        cond = {"<": DConsts.triglenLess, 
                ">": DConsts.triglenMore, 
                "=": DConsts.triglenTimeout}

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
        
        # Set trigger condition (rise/fall/either)
        dwf.FDwfAnalogInTriggerConditionSet(self.hdwf, polar[polarity])
        
        # Set pulse width and condition
        dwf.FDwfAnalogInTriggerLengthSet(self.hdwf, c_double(width))
        dwf.FDwfAnalogInTriggerLengthConditionSet(self.hdwf, cond[condition])
        

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
        self.update_data_connectors(df)

        # Update last_data for UI access
        self.last_data = df
        self.data_ready = True

        return df

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
        stack_df = []
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
                df = pd.DataFrame({
                'time': numpy.linspace(0, nSamples/frequency, nSamples),
                'ch1': numpy.frombuffer(channel1[i], dtype=numpy.float64),
                'ch2': numpy.frombuffer(channel2[i], dtype=numpy.float64)
                })
                stack_df.append(df)
                self.update_data_connectors(df)
                # Update last_data for UI access
                self.last_data = stack
                self.data_ready = True
        except KeyboardInterrupt:
            print("Keyboard Interrupt at Acquisition {}".format(i))
            return None
        return stack_df
    
    def acquire_continuous(self, callback=None, verbose=0):
        """
        Acquire data continuously from the device.
        
        This method sets up the device for continuous acquisition and calls the
        callback function after each acquisition."""
        nSamples = self.params["scope"]["samples"]
        frequency = self.params["scope"]["frequency"]
        # Initialize the stack

        channel1 = (c_double*nSamples)()
        channel2 = (c_double*nSamples)()
        
        # Initial configuration
        self.dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(1))
        
        # Acquire the stack
        print("Beginning continuous acquisition")
        try:
            while True:
                # Update capture index for UI
                self.capture_index += 1
                if verbose > 0:
                    display("Acquiring: {}".format(self.capture_index))
                    clear_output(wait=True)
                
                # Make sure we're properly configured for the next acquisition
                # Ensure trigger settings are properly applied
                self.configure_trigger(self.params["trigger"])
                
                # Start the acquisition
                self.dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(1))
                
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
                df = pd.DataFrame({
                'time': numpy.linspace(0, nSamples/frequency, nSamples),
                'ch1': numpy.frombuffer(channel1, dtype=numpy.float64),
                'ch2': numpy.frombuffer(channel2, dtype=numpy.float64)
                })
                self.update_data_connectors(df)
                # Update last_data for UI access
                self.last_data = df
                self.data_ready = True
                
                # Call the callback if provided
                if callback:
                    callback(df)
                    
                # Small sleep to prevent CPU overload
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("Keyboard Interrupt - Continuous acquisition stopped")
            return None

    def update_data_connectors(self, df):
        """
        Update the data connectors with the latest data.
        Thread-safe implementation that won't crash if UI is accessed from multiple threads.
        Decimates data to reduce the amount sent to the UI while preserving the overall waveform.
        """
        if self.data_connectors is not None:
            try:
                self.data_connectors[0].cb_set_data(df['ch1'], df['time'])
                self.data_connectors[1].cb_set_data(df['ch2'], df['time'])
            except Exception as e:
                # Safely handle errors that might occur due to threading issues
                print(f"Error updating data connectors: {e}")

    def graph(self):
        """Initialize UI on the main thread to ensure thread safety with PyQt."""
        # Ensure QApplication is created in the main thread
        app = get_qt_app()
        
        # Create UI on main thread
        self.ui = OscilloscopeUI(self, app, npoints = self.params["scope"]["samples"])
        
        # No need to start a thread for the event loop
        # The event loop should run in the main thread where the UI was created
        # The calling code should call app.exec_() after this
        
        return self.ui  # Return UI object for caller's reference

    def _run_event_loop(self, app):
        """
        Run the Qt event loop in a separate thread.
        Note: This is generally not recommended - the event loop should run in the main thread.
        """
        # This method is kept for backward compatibility but is not the preferred approach
        if threading.current_thread() is not threading.main_thread():
            app.exec_()

    def create_ui(self):
        """Create the UI - maintained for backward compatibility."""
        return self.graph()

    def set_data_connectors(self, data_connectors):
        self.data_connectors = data_connectors
    
    def __del__(self):
        self.close()
        

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

def ui_trigger_test(params):
    """Run a test with the oscilloscope UI in a thread-safe manner."""
    # Initialize Qt application in main thread
    app = get_qt_app()
    
    # Create DigiScope instance
    ds = DigiScope()
    ds.configure_all(params)
    
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

if __name__ == "__main__":
    #test with GUI
    frequency = 1e5  # Sampling frequency
    duration = 0.1   # Duration of capture
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
            "level": 0.5,        # Reduced trigger level for cleaner triggering
            "polarity": "+",     # Rising edge only
            "hysteresis": 0.1,   # Increased hysteresis to prevent double triggering
            "position": 0.2      # Position trigger point 20% from start
            #"width": 0.05,
            #"condition": ">",
        },
        "wavegen": {
            "waveform": "sine",
            "frequency": 15,      # Slower frequency for more reliable triggering
            "offset": 0.0,
            "amplitude": 1.5
        }
    }
    #self_trigger_test(custom_params)
    #stack_acquire_test(custom_params)
    ui_trigger_test(custom_params)