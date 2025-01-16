"""
BORIS
Behavioral Observation Research Interactive Software
Copyright 2012-2024 Olivier Friard


  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
  MA 02110-1301, USA.

"""

import logging
import sys
import mne
import pandas as pd

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import pyqtSignal, QEvent, QThread, QObject, pyqtSlot
from PyQt5.QtWidgets import (
    QSizePolicy,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QSpacerItem,
)


class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.axes = self.fig.add_subplot(1, 1, 1)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class Plot_data(QWidget):
    send_fig = pyqtSignal(float)

    # send keypress event to mainwindow
    sendEvent = pyqtSignal(QEvent)

    def __init__(
        self,
        file_name,
        xaxis_top_title,
        time_offset,
        plot_style,
        yaxis_title,
        y_label,
        columns_to_plot,
        xaxis_bottom_title,
        log_level="",
    ):
        super().__init__()

        self.installEventFilter(self)

        self.setWindowTitle(f"External data: {yaxis_title}")

        self.xaxis_bottom_title = xaxis_bottom_title

        self.myplot = MyMplCanvas(self)

        self.time_out = 10
        self.time_offset = float(time_offset)

        self.layout = QVBoxLayout()
        self.toolbar = NavigationToolbar(self.myplot, self)

        self.hlayout1 = QHBoxLayout()

        self.hlayout1.addWidget(QLabel("Red vertical lines denote 'Reach'. "))
        self.hlayout1.addWidget(QLabel("Green vertical lines denote 'Start of Rest'. "))

        self.hlayout1.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.hlayout2 = QHBoxLayout()
        self.hlayout2.addWidget(QLabel("File time in seconds:"))
        self.lb_value = QLabel("")
        self.hlayout2.addWidget(self.lb_value)
        self.hlayout2.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.layout.addLayout(self.hlayout1)
        self.layout.addLayout(self.hlayout2)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.myplot)

        self.setLayout(self.layout)

        self.resize(960, 540)

        self.plot_style = plot_style
        self.yaxis_title = yaxis_title

        self.xaxis_top_title = xaxis_top_title
        self.y_label = y_label
        self.error_msg = ""

        result = self._load_data(file_name, columns_to_plot)

        if not result:
            return
        
        # does not have to do with graph updates
        min_time_step = 50

        # plotter and thread are none at the beginning
        self.plotter = Plotter()
        self.plotter.data = self.data.to_numpy()  # Ensure data is in numpy format
        self.plotter.annotations = self.annotations
        self.plotter.max_frequency = 1

        self.plotter.min_value = 0
        self.plotter.max_value = 50

        self.plotter.min_time_value = 0
        self.plotter.max_time_value = 1000

        self.plotter.min_time_step = min_time_step

        self.thread = QThread()

        # connect signals
        self.send_fig.connect(self.plotter.replot)
        self.plotter.return_fig.connect(self.plot)
        # move to thread and start
        self.plotter.moveToThread(self.thread)
        self.thread.start()

        if min_time_step < 0.2:
            self.time_out = 200
        else:
            self.time_out = round(min_time_step * 1000)

    def eventFilter(self, receiver, event):
        """
        send event (if keypress) to main window
        """
        if event.type() == QEvent.KeyPress:
            self.sendEvent.emit(event)
            return True
        else:
            return False

    def timer_plot_data_out(self, time_):
        self.update_plot(time_)

    def update_plot(self, time_):
        """
        update plot by signal
        """
        self.send_fig.emit(float(time_) + float(self.time_offset))

    def close_plot(self):
        self.thread.quit()
        self.thread.wait()
        self.close()
    
    # Slot receives data and plots it
    def plot(self, x, y, position_data, position_start, min_value, max_value, position_end, annotations):
        # print current value
        try:
            self.lb_value.setText(str(position_data))
        except Exception:
            self.lb_value.setText("Read error")

        logging.debug(annotations)
        try:
            if not hasattr(self, 'main_plot_drawn') or not self.main_plot_drawn:
                self.myplot.axes.clear()
                self.myplot.axes.set_title(self.yaxis_title)
                self.myplot.axes.set_xlim(position_start, position_end)
                self.myplot.axes.set_ylabel(self.y_label, rotation=90, labelpad=2)
                self.myplot.axes.set_xlabel(self.xaxis_bottom_title, rotation=0, labelpad=2)
                self.myplot.axes.set_ylim((min_value, max_value))
                self.myplot.axes.plot(x, y, self.plot_style, linewidth = 0.2)
                self.black_line = self.myplot.axes.plot([(position_data + self.time_offset)*50, (position_data + self.time_offset)*50], [min_value, max_value], color='black', linewidth=2)[0]

                ax2 = self.myplot.axes.twiny()  # Create a second x-axis sharing the same y-axis
                ax2.set_xlim((position_start-self.time_offset), (position_end-(self.time_offset*50))/50)  # Ensure the second x-axis matches the first one
                ax2.set_xlabel(self.xaxis_top_title, rotation=0, labelpad=2)  # Label for the second x-axis

                # Plot vertical lines for each event (onset times) with different colors for each event type
                for onset, description in zip(annotations.onset, annotations.description):
                    if description == 'Reach':
                        self.myplot.axes.axvline(x=onset*50, color='red', linestyle='--', label='Reach')
                    elif description == 'Start of Rest':
                        self.myplot.axes.axvline(x=onset*50, color='green', linestyle=':', label='Start of Rest')

                self.myplot.figure.tight_layout()
                #self.myplot.axes.axvline(x=position_data, color=cfg.REALTIME_PLOT_CURSOR_COLOR, linestyle="-")

                self.main_plot_drawn = True
            else:
                self.black_line.set_xdata([(position_data + self.time_offset) * 50, (position_data + self.time_offset) * 50])
            
            self.myplot.draw()

        except Exception:
            logging.debug(f"error in plotting external data: {sys.exc_info()[1]}")

 
    def _load_data(self, file_name, columns_to_plot):
        """
        Load and process data, with specific handling for SNIRF files.
        """
        try:
            # Check if it's a SNIRF file and load it accordingly
            if file_name.endswith('.snirf'):
                self.snirf_data = mne.io.read_raw_snirf(file_name)

                # Log the basic info about SNIRF data
                logging.debug(f"SNIRF data info: {self.snirf_data.info}")

                # Check if self_snirf_data is valid
                if self.snirf_data is None or len(self.snirf_data.times) == 0:
                    self.error_msg = f"Error loading SNIRF file: {file_name}"
                    return 0

                raw_data = self.snirf_data.get_data()
                self.annotations = self.snirf_data.annotations
                channel_names = self.snirf_data.info['ch_names']

                # Log data shape and channel names
                logging.debug(f"Raw data shape: {raw_data.shape}")
                logging.debug(f"Channel names: {channel_names}")

                # Convert data into a pandas DataFrame
                self.data = pd.DataFrame(raw_data.T, columns=channel_names)

                # Log the first few rows of data to ensure it's loaded
                logging.debug(f"First few rows of data: {self.data.head()}")

                # Check if data is being populated correctly
                if self.data.empty:
                    logging.error("No data found in the loaded SNIRF file.")
                else:
                    logging.debug(f"Data loaded successfully, number of rows: {len(self.data)}")

                # # Apply converters and column filtering if necessary
                # if converters:
                #     self.data = self.data.apply(converters)
                #     logging.debug(f"Data after applying converters: {self.data.head()}")
                    
                # Handle the `columns_to_plot` input
                if columns_to_plot:
                    if isinstance(columns_to_plot, str):
                        columns_to_plot = [int(col.strip()) - 1 for col in columns_to_plot.split(',')]

                    # Map the columns_to_plot indices to the correct column names
                    columns_to_plot = [self.data.columns[i] for i in columns_to_plot if i < len(self.data.columns)]
                    logging.debug(f"Columns to plot: {columns_to_plot}")

                    # Filter the data to only include the columns to plot
                    self.data = self.data[columns_to_plot]
                    logging.debug(f"Data after applying columns_to_plot: {self.data.head()}")

            else:
                self.error_msg = "Unsupported file format. Only SNIRF files are supported."
                logging.error(self.error_msg)
                return 0

        except Exception as e:
            self.error_msg = f"Error processing file {file_name}: {str(e)}"
            logging.error(self.error_msg)
            return 0
        
        return 1

class Plotter(QObject):
    return_fig = pyqtSignal(
        np.ndarray,  # x array
        np.ndarray,  # y array
        float,  # position_data
        float,  # position start
        float,  # min value
        float,  # max value
        float,  # position end
        mne.Annotations, # events?
    )

    @pyqtSlot(float)
    def replot(self, current_time):  # time_ in s

        self.black_line = False

        logging.debug("current_time: {}".format(current_time))

        current_discrete_time = round(round(current_time / self.min_time_step) * self.min_time_step, 2)

        logging.debug("current_discrete_time: {}".format(current_discrete_time))
        # logging.debug("self.interval: {}".format(self.interval))

        # freq_interval = int(round(self.interval / self.min_time_step))

        annotations = self.annotations

        if self.min_time_value <= current_discrete_time <= self.max_time_value:
            logging.debug("self.min_time_value <= current_discrete_time <= self.max_time_value")
            logging.debug(f"Shape of self.data: {self.data.shape}")

            # Use the entire dataset for y (first column)
            y = self.data[:, 0]
            logging.debug(f"First 10 values of y: {y[:10]}")  # Log the first 10 values for sanity check
            logging.debug(f"Total length of y: {len(y)}")

        # elif current_time > self.max_time_value:
        #     logging.debug(f"self.interval/self.min_time_step/2: {self.interval / self.min_time_step / 2}")

        #     dim_footer = int(round((current_time - self.max_time_value) / self.min_time_step + self.interval / self.min_time_step / 2))

        #     footer = np.array([np.nan] * dim_footer).T
        #     logging.debug(f"len footer: {len(footer)}")

        #     a = (self.interval / 2 - (current_time - self.max_time_value)) / self.min_time_step
        #     logging.debug(f"a: {a}")

        #     if a >= 0:
        #         logging.debug("a>=0")

        #         st = int(round(len(self.data) - a))
        #         logging.debug(f"st: {st}")

        #         flag_i = False
        #         if st < 0:
        #             i = np.array([np.nan] * abs(st)).T
        #             st = 0
        #             flag_i = True

        #         y = np.append(self.data[st : len(self.data)][:, 0], footer, axis=0)

        #         if flag_i:
        #             y = np.append(i, y, axis=0)

        #         logging.debug(f"len y a>=0: {len(y)}")

        #     else:  # a < 0
        #         logging.debug("a<0")
        #         y = np.array([np.nan] * int(self.interval / self.min_time_step)).T

        #         logging.debug(f"len y a<0: {len(y)}")

        # elif current_time < self.min_time_value:
        #     x = (self.min_time_value - current_time) / self.min_time_step
        #     dim_header = int(round(self.interval / self.min_time_step / 2 + x))
        #     header = np.array([np.nan] * dim_header).T

        #     b = int(round(self.interval / self.min_time_step / 2 - x))

        #     if b >= 0:
        #         y = np.append(header, self.data[0:b][:, 0], axis=0)
        #         if len(y) < freq_interval:
        #             y = np.append(y, np.array([np.nan] * int(freq_interval - len(y))).T, axis=0)

        #     else:
        #         y = np.array([np.nan] * int(self.interval / self.min_time_step)).T

        logging.debug(f"len y after adjustments: {len(y)}")
        logging.debug(f"First 10 values of y after adjustments: {y[:10]}")  # Log adjusted values

        # Modify x to be the row indices of self.data (0 to len(self.data)-1)
        x = np.arange(len(self.data))
        logging.debug(f"First 10 values of x: {x[:10]}")  # Log the first 10 values of x
        logging.debug(f"Total length of x: {len(x)}")

        # Ensure that x and y have the same length before passing to plotting function
        if len(x) != len(y):
            logging.error(f"Error: x and y have different lengths. x: {len(x)}, y: {len(y)}")
            return

        self.min_value = y.min()
        self.max_value = y.max()
        # Emit the plot data
        self.return_fig.emit(
            x,
            y,
            current_time,  # position_data
            x.min(),
            #current_discrete_time - self.interval // 2,  # position_start
            self.min_value,
            self.max_value,
            x.max(),
            annotations,
            #current_discrete_time + self.interval // 2,  # position_end,
        )
