#imports
import sys
try:
    # Try to import from PyQt5 first
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QLineEdit, QGroupBox, QFormLayout
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QPen, QColor
except ImportError:
    # Fall back to PyQt6 if PyQt5 is not available or incompatible
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QLineEdit, QGroupBox, QFormLayout
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtGui import QPen, QColor
import pyqtgraph as pg
from pglive.sources.data_connector import DataConnector
from threading import Thread, Lock
import json 
from time import sleep
from math import sin
from pglive.sources.live_plot import LiveLinePlot
from pglive.sources.live_plot_widget import LivePlotWidget
import datetime
import numpy as np
import ipywidgets as widgets

# Global variable to hold QApplication instance
_app = None


def get_qt_app():
    """Returns the global QApplication instance, creating it if necessary."""
    global _app
    if _app is None:
        _app = QApplication.instance()
        if _app is None:
            _app = QApplication(sys.argv)
    return _app

class SettingTable():
    """UI element for jupyter notebook to display and edit oscilloscope settings,
    read/write to json file
    call:
    table = SettingTable(scope)
    table.display()
    """
    def __init__(self, scope, params_path=None):
        self.scope = scope
        self.submit_func = scope.configure_all
        self.save_func = scope.save_params
        self.params_path = params_path
        self.params = scope.params
        self.param_widgets = {}
        self.tab = widgets.Tab()
        self.children = []
        self.tab_titles = []
        
    def unpile_params(self):
        """Generate widget tree from the params dictionary"""
        params = self.scope.params
        for section_key, section_value in params.items():
            section_widgets = []
            for param_key, param_value in section_value.items():
                if isinstance(param_value, bool):
                    widget = widgets.Checkbox(value=param_value, description=param_key)
                elif isinstance(param_value, (int, float)):
                    widget = widgets.FloatText(value=param_value, description=param_key)
                else:
                    widget = widgets.Text(value=str(param_value), description=param_key)
                section_widgets.append(widget)
            self.children.append(widgets.VBox(section_widgets))
            self.tab_titles.append(str(section_key))
        
    def pile_params(self):
        """Generate new params dictionary from the  widget values"""
        params = {}
        for section_key, section_value in self.param_widgets.items():
            params[section_key] = {}
            for param_key, widget in section_value.items():
                params[section_key][param_key] = widget.value
        self.params = params
        return params
    
    def display(self):
        """display the setting table in jupyter notebook"""
        self.unpile_params()
        
        # Set the tab children and titles
        self.tab.children = self.children
        for i, title in enumerate(self.tab_titles):
            self.tab.set_title(i, title)
        
        # Create buttons for actions
        self.save_button = widgets.Button(description="Save")
        self.apply_button = widgets.Button(description="Apply")
        
        # Define button callbacks
        self.save_button.on_click(self.on_save_clicked)
        self.apply_button.on_click(self.on_apply_clicked)
        
        # Create a button box
        self.button_box = widgets.HBox([self.apply_button, self.save_button])
        
    def on_save_clicked(self, b):
        """Save the current params dictionary to the json file"""
        self.pile_params()
        self.scope.save_params(self.params_path)
            
    def on_apply_clicked(self, b):
        """Regenerate params dictionary from the widget values, call
        configure_all to apply to scope"""
        self.pile_params()
        self.scope.configure_all(self.params)
    
class OscilloscopeUI(QMainWindow):
    """usage:
    scope.ui = OscilloscopeUI(scope)
    then the data connector gets assigned to scope,
    and all data acq will call the data connector
    """
    def __init__(self, scope, app=None, npoints=1000, max_plot_pts = 2000):
        super().__init__()
        # Use provided app or get the global one
        self.max_plot_pts = max_plot_pts
        self.ds = int(npoints / self.max_plot_pts)
        self.app = app if app is not None else get_qt_app()
        self.scope = scope
        self.lock = Lock()
        self.setup_ui_elements()
        
        # Configure data connectors with proper settings based on the pglive implementation
        # - update_rate: Controls how frequently new data is accepted (in Hz)
        # - plot_rate: Controls how frequently the plot is actually updated (in Hz)
        # - max_points: Controls maximum number of points stored
        self.data_connectors = [
            DataConnector(self.ch1_plot, update_rate=10, plot_rate=30, ignore_auto_range=True),
            DataConnector(self.ch2_plot, update_rate=10, plot_rate=30, ignore_auto_range=True)
        ]

        # Fix timestamp update issue by directly connecting to signals
        for connector in self.data_connectors:
            # Connect to the signal that's emitted when data is updated
            connector.sig_new_data.connect(lambda *args: self.update_timestamp())

        scope.set_data_connectors(self.data_connectors)
        self.running = False
        self.start()

    def update_Npoints(self, Npoints):
        """Update the number of points and downsampling in the data connectors
        
        This updates:
        1. max_points in the data connectors
        2. downsampling rate in the plot widgets
        3. Refreshes the data visualization
        """
        # Update downsampling rate - higher points = higher ds value
        self.ds = max(1, int(Npoints / self.max_plot_pts))
        
        # Update plot widget downsampling settings
        for plot_widget in [self.plot_widget_ch1, self.plot_widget_ch2]:
            plot_widget.setDownsampling(ds=self.ds, auto=True, mode='mean')
        
        # Store current data to restore after resizing
        with self.lock:
            # Extract current data from the connectors
            data_x = [list(conn.x) for conn in self.data_connectors]
            data_y = [list(conn.y) for conn in self.data_connectors]
            
            # Update the max_points property in the connectors
            for connector in self.data_connectors:
                connector.max_points = Npoints
            
            # Restore data if there was any
            for i, connector in enumerate(self.data_connectors):
                if data_x[i] and data_y[i]:
                    # Use cb_set_data to restore the data with new buffer sizes
                    connector.cb_set_data(data_y[i], data_x[i])

    def update_timestamp(self):
        """Update the timestamp label with the current time"""
        current_time = datetime.datetime.now()
        
        # Calculate time difference in seconds
        if not hasattr(self, 'last_timestamp'):
            self.last_timestamp = current_time
            time_diff_sec = 0
        else:
            # Calculate time difference in seconds
            time_diff_sec = (current_time - self.last_timestamp).total_seconds()
            
        # Format for display
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
        self.timestamp_label.setText(f"Last Update: {formatted_time} ({time_diff_sec:.3f}s)")
        
        # Update the last timestamp after displaying the difference
        self.last_timestamp = current_time

    def setup_ui_elements(self):
        """Setup. two charts with a shared x axis, ch1,ch2"""
        self.setWindowTitle("Diligent Digilent Grapher")
        self.setGeometry(100, 100, 800, 600)
        
        # Create a QVBoxLayout
        layout = QVBoxLayout()
        
        # Add timestamp label at the top
        self.timestamp_label = QLabel("Last Update: --")
        self.timestamp_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.timestamp_label)
        
        # Create two plot widgets with pglive
        self.plot_widget_ch1 = LivePlotWidget(title="Channel 1")
        self.plot_widget_ch2 = LivePlotWidget(title="Channel 2")
        
        # Configure plot widgets for smoother data visualization
        for plot_widget in [self.plot_widget_ch1, self.plot_widget_ch2]:
            # Enable anti-aliasing for smoother line rendering
            plot_widget.setAntialiasing(True)
            # Let pyqtgraph handle downsampling automatically
            plot_widget.setDownsampling(ds=self.ds, auto=True, mode='mean')
            # Only render what's visible in the view
            plot_widget.setClipToView(True)
            # Add grid for better readability
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            # Configure view box
            view_box = plot_widget.getPlotItem().getViewBox()
            # Enable auto-range but optimize for performance
            view_box.enableAutoRange(x=True, y=True)
            view_box.setAutoVisible(y=True)
            # Set proper mouse interaction mode
            view_box.setMouseMode(view_box.RectMode)
            # Improve performance with limited range updates
            #sview_box.setLimits(yMin=-10, yMax=10)
        
        # Create the plots with smoother line rendering
        #from PyQt5.QtCore import Qt
        yellow_pen = QPen(QColor(255, 255, 0))
        yellow_pen.setWidth(0)  # Slightly thicker for smoother appearance
        try:
            # Qt5 style
            yellow_pen.setStyle(Qt.SolidLine)
        except AttributeError:
            # Qt6 style
            yellow_pen.setStyle(Qt.PenStyle.SolidLine)
        
        blue_pen = QPen(QColor(0, 0, 255))
        blue_pen.setWidth(0)  # Slightly thicker for smoother appearance
        try:
            # Qt5 style
            blue_pen.setStyle(Qt.SolidLine)
        except AttributeError:
            # Qt6 style
            blue_pen.setStyle(Qt.PenStyle.SolidLine)
        
        # Create plots with proper connection mode for smooth lines
        self.ch1_plot = LiveLinePlot(pen=yellow_pen, connect="all")
        self.ch2_plot = LiveLinePlot(pen=blue_pen, connect="all")
        
        # Add each plot to its respective widget
        self.plot_widget_ch1.addItem(self.ch1_plot)
        self.plot_widget_ch2.addItem(self.ch2_plot)
        
        # Link X axes of both plots
        self.plot_widget_ch2.setXLink(self.plot_widget_ch1)
        
        # Add plot widgets to layout vertically
        layout.addWidget(self.plot_widget_ch1)
        layout.addWidget(self.plot_widget_ch2)
        
        # Add Y-range controls in a more compact layout
        y_range_layout = QHBoxLayout()
        
        # Channel 1 Y-range controls
        ch1_group = QGroupBox("CH1 Y-Range")
        ch1_group.setFixedHeight(70)
        ch1_layout = QHBoxLayout()
        ch1_layout.setContentsMargins(5, 2, 5, 2)
        ch1_layout.setSpacing(5)
        
        # Create min/max containers for CH1
        ch1_min_layout = QHBoxLayout()
        ch1_min_layout.setSpacing(2)
        ch1_min_layout.addWidget(QLabel("Min:"))
        self.ch1_ymin = QLineEdit("-5")
        self.ch1_ymin.setFixedWidth(50)
        self.ch1_ymin.returnPressed.connect(lambda: self.set_y_range(0))
        ch1_min_layout.addWidget(self.ch1_ymin)
        
        ch1_max_layout = QHBoxLayout()
        ch1_max_layout.setSpacing(2)
        ch1_max_layout.addWidget(QLabel("Max:"))
        self.ch1_ymax = QLineEdit("5")
        self.ch1_ymax.setFixedWidth(50)
        self.ch1_ymax.returnPressed.connect(lambda: self.set_y_range(0))
        ch1_max_layout.addWidget(self.ch1_ymax)
        
        # Add min/max containers to main layout
        ch1_layout.addLayout(ch1_min_layout)
        ch1_layout.addLayout(ch1_max_layout)
        ch1_group.setLayout(ch1_layout)
        
        # Channel 2 Y-range controls
        ch2_group = QGroupBox("CH2 Y-Range")
        ch2_group.setFixedHeight(70)
        ch2_layout = QHBoxLayout()
        ch2_layout.setContentsMargins(5, 2, 5, 2)
        ch2_layout.setSpacing(5)
        
        # Create min/max containers for CH2
        ch2_min_layout = QHBoxLayout()
        ch2_min_layout.setSpacing(2)
        ch2_min_layout.addWidget(QLabel("Min:"))
        self.ch2_ymin = QLineEdit("-5")
        self.ch2_ymin.setFixedWidth(50)
        self.ch2_ymin.returnPressed.connect(lambda: self.set_y_range(1))
        ch2_min_layout.addWidget(self.ch2_ymin)
        
        ch2_max_layout = QHBoxLayout()
        ch2_max_layout.setSpacing(2)
        ch2_max_layout.addWidget(QLabel("Max:"))
        self.ch2_ymax = QLineEdit("5")
        self.ch2_ymax.setFixedWidth(50)
        self.ch2_ymax.returnPressed.connect(lambda: self.set_y_range(1))
        ch2_max_layout.addWidget(self.ch2_ymax)
        
        # Add min/max containers to main layout
        ch2_layout.addLayout(ch2_min_layout)
        ch2_layout.addLayout(ch2_max_layout)
        ch2_group.setLayout(ch2_layout)
        
        # Add groups to layout
        y_range_layout.addWidget(ch1_group)
        y_range_layout.addWidget(ch2_group)
        layout.addLayout(y_range_layout)
        
        # Create a single reset button that resets both views
        reset_button = QPushButton("Reset Views")
        reset_button.setFixedWidth(100)  # Make button small and fixed width
        reset_button.clicked.connect(self.reset_all_views)
        
        # Create a horizontal layout to center the button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(reset_button)
        button_layout.addStretch()
        
        # Add button layout below the graphs
        layout.addLayout(button_layout)
        
        # Create central widget to hold the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def start(self):
        self.show()  # Show the main window instead of just the plot widget
        self.running = True

    def stop(self):
        with self.lock:
            self.scope.data_connectors = None
            self.running = False
            
    def reset_plots(self):
        """Reset the plots by recreating the data connectors"""
        # Create new data connectors with the same optimized settings
        self.data_connectors = [
            DataConnector(self.ch1_plot, max_points=self.max_plot_pts, update_rate=5, plot_rate=30, ignore_auto_range=True),
            DataConnector(self.ch2_plot, max_points=self.max_plot_pts, update_rate=5, plot_rate=30, ignore_auto_range=True)
        ]
        
        # Fix timestamp update issue by directly connecting to signals in reset_plots too
        for connector in self.data_connectors:
            # Connect to the signal that's emitted when data is updated
            connector.sig_new_data.connect(lambda *args: self.update_timestamp())
        
        # Reapply the pen configuration for smooth rendering
        # Get the appropriate Qt constant based on whether we're using PyQt5 or PyQt6
        try:
            # Qt5 style
            solid_line = Qt.SolidLine
        except AttributeError:
            # Qt6 style
            solid_line = Qt.PenStyle.SolidLine
            
        yellow_pen = QPen(QColor(255, 255, 0))
        yellow_pen.setWidth(0)  # Slightly thicker for smoother appearance

        yellow_pen.setStyle(solid_line)

        
        blue_pen = QPen(QColor(0, 0, 255))
        blue_pen.setWidth(0)  # Slightly thicker for smoother appearance
        blue_pen.setStyle(solid_line)
     
        
        self.ch1_plot.setPen(yellow_pen)
        self.ch2_plot.setPen(blue_pen)
        
        # Update the scope with the new connectors
        self.scope.set_data_connectors(self.data_connectors)

    def reset_all_views(self):
        """Reset the view limits for both channels"""
        # Completely reset view for channel 1
        view_box1 = self.plot_widget_ch1.getPlotItem().getViewBox()
        view_box1.enableAutoRange(x=True, y=True)
        view_box1.setAutoVisible(x=True, y=True)
        view_box1.autoRange()  # Force immediate auto-range

        # Completely reset view for channel 2
        view_box2 = self.plot_widget_ch2.getPlotItem().getViewBox()
        view_box2.enableAutoRange(x=True, y=True)
        view_box2.setAutoVisible(x=True, y=True)
        view_box2.autoRange()  # Force immediate auto-range

    def set_y_range(self, channel):
        """Set the Y range for a specific channel"""
        try:
            if channel == 0:
                ymin = float(self.ch1_ymin.text())
                ymax = float(self.ch1_ymax.text())
                view_box = self.plot_widget_ch1.getPlotItem().getViewBox()
            else:
                ymin = float(self.ch2_ymin.text())
                ymax = float(self.ch2_ymax.text())
                view_box = self.plot_widget_ch2.getPlotItem().getViewBox()
            
            # Ensure min is less than max
            if ymin >= ymax:
                raise ValueError("Min must be less than Max")
                
            # Set Y range and disable auto-range
            view_box.setYRange(ymin, ymax)
            view_box.enableAutoRange(y=False)
        except ValueError as e:
            # In a real application, you might want to show an error dialog here
            print(f"Invalid Y range: {e}")


# Test code to run the UI with simulated data
if __name__ == "__main__":
    class SimulatedScope:
        """Simple class to simulate a scope for testing the UI"""
        def __init__(self):
            self.data_connectors = None
            self.x = 0
            self.timer = QTimer()
            self.timer.timeout.connect(self.generate_data)
            
        def set_data_connectors(self, connectors):
            self.data_connectors = connectors
            # Start generating data when connectors are set
            self.timer.start(50)  # Update more frequently (every 50ms) to test timestamp updating
            
        def generate_data(self):
            if self.data_connectors is None:
                return
                
            # Generate some sample data - sine waves with different frequencies
            x_values = np.linspace(self.x, self.x + 0.1, 10)
            ch1_data = np.sin(2 * np.pi * 1.0 * x_values)
            ch2_data = 0.5 * np.sin(2 * np.pi * 2.0 * x_values + 0.5)
            
            # Push data to both channels
            for i, x in enumerate(x_values):
                if self.data_connectors[0]:
                    self.data_connectors[0].cb_append_data_point(y=ch1_data[i], x=x)
                if self.data_connectors[1]:
                    self.data_connectors[1].cb_append_data_point(y=ch2_data[i], x=x)
            
            self.x += 0.1
    
    # Create the simulated scope
    scope = SimulatedScope()
    
    # Create the UI
    app = get_qt_app()
    ui = OscilloscopeUI(scope)
    
    # Start the Qt event loop
    sys.exit(app.exec_())


