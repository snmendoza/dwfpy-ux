#imports
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPen, QColor
import pyqtgraph as pg
from pglive.sources.data_connector import DataConnector
from threading import Thread, Lock
from time import sleep
from math import sin
from pglive.sources.live_plot import LiveLinePlot
from pglive.sources.live_plot_widget import LivePlotWidget


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


class OscilloscopeUI(QMainWindow):
    """usage:
    scope.ui = OscilloscopeUI(scope)
    then the data connector gets assigned to scope,
    and all data acq will call the data connector
    """
    def __init__(self, scope, app=None, npoints=1000):
        super().__init__()
        # Use provided app or get the global one
        self.ds = int(npoints / 1000)
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

        scope.set_data_connectors(self.data_connectors)
        self.running = False
        self.start()

    def setup_ui_elements(self):
        """Setup. two charts with a shared x axis, ch1,ch2"""
        self.setWindowTitle("Diligent Digilent Grapher")
        self.setGeometry(100, 100, 800, 600)
        
        # Create a QVBoxLayout
        layout = QVBoxLayout()
        
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
            view_box.setLimits(yMin=-10, yMax=10)
        
        # Create the plots with smoother line rendering
        from PyQt5.QtCore import Qt
        yellow_pen = QPen(QColor(255, 255, 0))
        yellow_pen.setWidth(0)  # Slightly thicker for smoother appearance
        yellow_pen.setStyle(Qt.SolidLine)
        
        blue_pen = QPen(QColor(0, 0, 255))
        blue_pen.setWidth(0)  # Slightly thicker for smoother appearance
        blue_pen.setStyle(Qt.SolidLine)
        
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
            DataConnector(self.ch1_plot, max_points=1000, update_rate=5, plot_rate=30, ignore_auto_range=True),
            DataConnector(self.ch2_plot, max_points=1000, update_rate=5, plot_rate=30, ignore_auto_range=True)
        ]
        
        # Reapply the pen configuration for smooth rendering
        from PyQt5.QtCore import Qt
        yellow_pen = QPen(QColor(255, 255, 0))
        yellow_pen.setWidth(0)  # Slightly thicker for smoother appearance
        yellow_pen.setStyle(Qt.SolidLine)
        
        blue_pen = QPen(QColor(0, 0, 255))
        blue_pen.setWidth(0)  # Slightly thicker for smoother appearance
        blue_pen.setStyle(Qt.SolidLine)
        
        self.ch1_plot.setPen(yellow_pen)
        self.ch2_plot.setPen(blue_pen)
        
        # Update the scope with the new connectors
        self.scope.set_data_connectors(self.data_connectors)


