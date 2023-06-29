# -*- coding: utf-8 -*-
import json
import math
import os
import re

import brightway2 as bw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PySide2 import QtWidgets, QtCore, QtWebEngineWidgets, QtWebChannel
from PySide2.QtCore import QObject, Slot
from PySide2.QtWidgets import QMenu, QAction, QApplication
import bokeh.layouts
from bokeh.embed import file_html
from bokeh.io import export_png, export_svg
from bokeh.models import (ColumnDataSource, HoverTool, CustomJS, Span, WheelZoomTool, Whisker, Label, BasicTicker,
                    PrintfTickFormatter, Tooltip
                    )
from bokeh.models.dom import HTML
from bokeh.palettes import turbo, magma
from bokeh.plotting import figure as bokeh_figure
from bokeh.transform import dodge, transform, linear_cmap
from bw2data.filesystem import safe_filename
from jinja2 import Template
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from numpy import mgrid

from activity_browser.ui.web import webutils
from .. import utils
from ..bwutils.commontasks import wrap_text, wrap_text_by_separator
from ..settings import ab_settings


class Plot(QtWidgets.QWidget):
    ALL_FILTER = "All Files (*.*)"
    PNG_FILTER = "PNG (*.png)"
    SVG_FILTER = "SVG (*.svg)"

    def __init__(self, parent=None):
        super().__init__(parent)
        # create figure, canvas, and axis
        # self.figure = Figure(tight_layout=True)
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)  # create an axis
        self.plot_name = 'Figure'

        # set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def reset_plot(self) -> None:
        self.figure.clf()
        self.ax = self.figure.add_subplot(111)

    def get_canvas_size_in_inches(self):
        # print("Canvas size:", self.canvas.get_width_height())
        return tuple(x / self.figure.dpi for x in self.canvas.get_width_height())

    def savefilepath(self, default_file_name: str, file_filter: str = ALL_FILTER):
        default = default_file_name or "LCA results"
        safe_name = safe_filename(default, add_hash=False)
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent=self,
            caption='Choose location to save lca results',
            dir=os.path.join(ab_settings.data_dir, safe_name),
            filter=file_filter,
        )
        return filepath

    def to_png(self):
        """ Export to .png format. """
        filepath = self.savefilepath(default_file_name=self.plot_name, file_filter=self.PNG_FILTER)
        if filepath:
            if not filepath.endswith('.png'):
                filepath += '.png'
            self.figure.savefig(filepath)

    def to_svg(self):
        """ Export to .svg format. """
        filepath = self.savefilepath(default_file_name=self.plot_name, file_filter=self.SVG_FILTER)
        if filepath:
            if not filepath.endswith('.svg'):
                filepath += '.svg'
            self.figure.savefig(filepath)


class LCAResultsPlot(Plot):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_name = 'LCA heatmap'

    def plot(self, df: pd.DataFrame):
        """ Plot a heatmap grid of the different impact categories and reference flows. """
        # need to clear the figure and add axis again
        # because of the colorbar which does not get removed by the ax.clear()
        self.reset_plot()

        dfp = df.copy()
        dfp.index = dfp['index']
        dfp.drop(dfp.select_dtypes(['object']), axis=1, inplace=True)  # get rid of all non-numeric columns (metadata)
        if "amount" in dfp.columns:
            dfp.drop(["amount"], axis=1, inplace=True)  # Drop the 'amount' col
        if 'Total' in dfp.index:
            dfp.drop("Total", inplace=True)

        # avoid figures getting too large horizontally
        dfp.index = [wrap_text(i, max_length=40) for i in dfp.index]
        dfp.columns = [wrap_text(i, max_length=20) for i in dfp.columns]

        sns.heatmap(
            dfp, ax=self.ax, cmap="Blues", annot=True, linewidths=0.05,
            annot_kws={"size": 11 if dfp.shape[1] <= 8 else 9,
                       "rotation": 0 if dfp.shape[1] <= 8 else 60}
        )
        self.ax.tick_params(labelsize=8)
        if dfp.shape[1] > 5:
            self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation="vertical")
        self.ax.set_yticklabels(self.ax.get_yticklabels(), rotation="horizontal")

        # refresh canvas
        size_inches = (2 + dfp.shape[0] * 0.5, 4 + dfp.shape[0] * 0.55)
        self.figure.set_size_inches(self.get_canvas_size_in_inches()[0], size_inches[1])
        size_pixels = self.figure.get_size_inches() * self.figure.dpi
        self.setMinimumHeight(size_pixels[1])

        self.canvas.draw()


class CorrelationPlot(Plot):
    def __init__(self, parent=None):
        super().__init__(parent)
        sns.set(style="darkgrid")

    def plot(self, df: pd.DataFrame):
        """ Plot a heatmap of correlations between different reference flows. """
        # need to clear the figure and add axis again
        # because of the colorbar which does not get removed by the ax.clear()
        self.reset_plot()
        canvas_size = self.canvas.get_width_height()
        # print("Canvas size:", canvas_size)
        size = (4 + df.shape[1] * 0.3, 4 + df.shape[1] * 0.3)
        self.figure.set_size_inches(size[0], size[1])

        corr = df.corr()
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        # Draw the heatmap with the mask and correct aspect ratio
        vmax = np.abs(corr.values[~mask]).max()
        # vmax = np.abs(corr).max()
        sns.heatmap(corr, mask=mask, cmap=plt.cm.PuOr, vmin=-vmax, vmax=vmax,
                    square=True, linecolor="lightgray", linewidths=1, ax=self.ax)

        df_lte8_cols = df.shape[1] <= 8
        for i in range(len(corr)):
            self.ax.text(
                i + 0.5, i + 0.5, corr.columns[i], ha="center", va="center",
                rotation=0 if df_lte8_cols else 45, size=11 if df_lte8_cols else 9
            )
            for j in range(i + 1, len(corr)):
                s = "{:.3f}".format(corr.values[i, j])
                self.ax.text(
                    j + 0.5, i + 0.5, s, ha="center", va="center",
                    rotation=0 if df_lte8_cols else 45, size=11 if df_lte8_cols else 9
                )
        self.ax.axis("off")

        # refresh canvas
        size_pixels = self.figure.get_size_inches() * self.figure.dpi
        self.setMinimumHeight(size_pixels[1])
        self.canvas.draw()


class SimpleDistributionPlot(Plot):
    def plot(self, data: np.ndarray, mean: float, label: str = "Value"):
        self.reset_plot()
        try:
            sns.histplot(data.T, kde=True, stat="density", ax=self.ax, edgecolor="none")
        except RuntimeError as e:
            print("Runtime error: {}\nPlotting without KDE.".format(e))
            sns.histplot(data.T, kde=False, stat="density", ax=self.ax, edgecolor="none")
        self.ax.set_xlabel(label)
        self.ax.set_ylabel("Probability density")
        # Add vertical line at given mean of x-axis
        self.ax.axvline(mean, label="Mean / amount", c="r", ymax=0.98)
        self.ax.legend(loc="upper right")
        _, height = self.canvas.get_width_height()
        self.setMinimumHeight(height / 2)
        self.canvas.draw()


class BokehPlot(QtWidgets.QWidget):
    """
    Author: Nabil Ahmed & Jonathan Kidner
    Class: BokehPlot

    Contains the utility functions for "publishing" and exporting bokeh figures for the AB project.
    Includes a non implemented plot() method that is used to generate the plots in child classes,
    NOTE this class is intended as a base class to provide common functionality for all derived
    Bokeh plotting classes.

    Utility functions include:
     - creating html page
     - determining figure height and width
     - legend style
     - axis style
     - colour style
     - resizing options
     - export functionality (png, svg)

    Inherited methods needing implementation:
     - plot(*args, **kwargs)
     - onContextMenu() # where applicable
    """
    #    BOKEH_JS_File_Name = "bokeh-2.4.1.min.js"
    #BOKEH_JS_File_Name = "bokeh-3.0.3.min.js"
    BOKEH_JS_File_Name = "bokeh-3.1.1.min.js"

    REST_COLOR = "#808080"
    ALL_FILTER = "All Files (*.*)"
    PNG_FILTER = "PNG (*.png)"
    SVG_FILTER = "SVG (*.svg)"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = None
        self.plot_name = 'Figure'

        self.view = QtWebEngineWidgets.QWebEngineView()
        self.page = QtWebEngineWidgets.QWebEnginePage()
        self.view.setContentsMargins(0, 0, 0, 0)

        # set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.view)

        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()
        self.size = QApplication.primaryScreen().size()

    ###### Formally part of the BokehPlotUtils class until otherwise specified
    # These methods were used throughout the Bokeh classes, hence their definition
    # in an external class while not complicating code legibility did not serve a
    # clear purpose.
    # Defining the utility methods within the base class, allows us to improve
    # functionality by increasing the polymorphism of the code, following OO
    # principles

    @staticmethod
    def build_html_bokeh_template(add_context_menu_communication: bool = False,
                                   disable_horizontal_scroll: bool = False):
        bokeh_jspath = webutils.get_static_js_path(BokehPlot.BOKEH_JS_File_Name)
        bokeh_js_code = utils.read_file_text(bokeh_jspath)
        template = Template("""
            <!DOCTYPE html>
            <html lang="en">
                <head>
                     <meta charset="utf-8">
                     <script type="text/javascript">""" + bokeh_js_code + """</script>
                     """ + (
        """<script src="qrc:///qtwebchannel/qwebchannel.js"></script>""" if add_context_menu_communication else "") + """
                    <link rel="stylesheet" href="../static/css/main.css"/>
                </head>
                    <body""" + (
                            """ style="overflow-x:hidden;" """ if disable_horizontal_scroll else "") + """>
                    {{ plot_div | safe }}
                    {{ plot_script | safe }}
                    """ + (
                            """<script type="text/javascript">
                                // Connect to QWebChannel and accept the injected chartBridge object for communication
                                // with Pyside 
                                new QWebChannel(qt.webChannelTransport, function (channel) {
                                    window.chartBridge = channel.objects.chartBridge;
                                });

                              // Called when user right-clicks, passes the co-ordinates to python via the injected  
                              // chartBridge object to open context menu from pyside
                              document.addEventListener('contextmenu', function(e) {
                                 window.chartBridge.set_context_menu_coordinates(JSON.stringify({x:window.lastHover.x, y: window.lastHover.y}));
                              }, true);
                                </script>""" if add_context_menu_communication else "") + """
                    </body>
                </html> """)
        return template

    @staticmethod
    def calculate_bar_chart_height(bar_count: int = 1, legend_item_count: int = 1):
        return 90 + (35 * bar_count) + (17 * legend_item_count)

    @staticmethod
    def calculate_results_chart_height(bar_count: int = 1, legend_item_count: int = 1):
            return 120 + (35 * bar_count) + (40 * legend_item_count)

    @staticmethod
    def style_and_place_legend(plot, location):
        new_legend = plot.legend[0]
        new_legend.location = location
        plot.legend[0] = None
        plot.legend[0].label_text_font_size = "10pt"
        plot.legend[0].label_text_font_style = "bold"
        plot.legend[0].label_height = 10
        plot.legend[0].label_standoff = 2
        plot.legend[0].glyph_width = 15
        plot.legend[0].glyph_height = 15
        plot.legend[0].spacing = 1
        plot.legend[0].margin = 0
        plot.legend.border_line_color = None
        new_legend.click_policy = 'hide'

    #        plot.add_layout(new_legend, 'below')

    @staticmethod
    def style_axis_labels(axis):
        axis.major_label_text_font_size = "10pt"
        axis.major_label_text_font_style = "bold"
        axis.major_label_text_line_height = 0.8
        axis.major_label_text_align = "right"

    @staticmethod
    def get_color_palette(length: int, first_grey: bool = False) -> tuple:
        base_colors = list(turbo(length if length < 256 else 256))
        if length > 256:
            backup_colors = list(magma(length - 256))
            base_colors = base_colors + backup_colors
        if first_grey:
            base_colors[0] = BokehPlot.REST_COLOR
        return tuple(base_colors)

    @staticmethod
    def reformat_labels(labels: list, limit: int = 50) -> list:
        newline = re.compile('(.*[$|\s])')
        formatted_labels = []
        for label in labels:
            location = 0
            new_label = []
            while location+limit < len(label):
                wrap = label[location:(limit+location)]
                m = newline.search(wrap)
                location = location + m.end()
                new_label.append(m.group(0))
            new_label.append(label[location:])
            label = "\n".join(new_label)
            formatted_labels.append(label)
        return formatted_labels

    @staticmethod
    def shorten_labels(labels: list, limit: int =48, indicator: str='..') -> list:
        formatted_labels = []
        for label in labels:
            if len(label) > (limit + len(indicator)):
                formatted_labels.append(label[:limit] + indicator)
            else:
                formatted_labels.append(label)
        return formatted_labels

    @staticmethod
    def resize_plot(self) -> None:
        pass
    # end of what was the formally BokehPlotUtils class

    def on_context_menu(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def reset_plot(self) -> None:
        pass

    def save_file_path(self, default_file_name: str, file_filter: str = ALL_FILTER):
        default = default_file_name or "LCA results"
        safe_name = safe_filename(default, add_hash=False)
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent=self,
            caption='Choose location to save lca results',
            dir=os.path.join(ab_settings.data_dir, safe_name),
            filter=file_filter,
        )
        return filepath

    def to_png(self):
        """ Export to .png format. """
        filepath = self.save_file_path(default_file_name=self.plot_name, file_filter=self.PNG_FILTER)
        if filepath:
            if not filepath.endswith('.png'):
                filepath += '.png'
            fig_width = self.figure.width + 500
            export_png(self.figure, filename=filepath, width=fig_width)

    def to_svg(self):
        """ Export to .svg format. """
        filepath = self.save_file_path(default_file_name=self.plot_name, file_filter=self.SVG_FILTER)
        if filepath:
            if not filepath.endswith('.svg'):
                filepath += '.svg'
            fig_width = self.figure.width + 500
            self.figure.output_backend = "svg"
            export_svg(self.figure, filename=filepath, width=fig_width)
            self.figure.output_backend = "canvas"


class LCAResultsBarChart(BokehPlot):
    """" Generate a bar chart comparing the absolute LCA scores of the products """
    BAR_HEIGHT = 0.7

    def on_context_menu(self, *args, **kwargs):
        raise NotImplementedError

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_name = 'LCA scores'

    def plot(self, df: pd.DataFrame, method: tuple, labels: list):
        df.index = pd.Index(labels)  # Replace index of tuples
        show_legend = df.shape[1] != 1  # Do not show the legend for 1 column
        df = df[::-1]

        if show_legend:
            column_source = ColumnDataSource(df)
        else:
            column_source = ColumnDataSource({'values': list(df[0].values), 'index': list(df.index)})

        x_max = max(df.max())
        x_min = min(df.min())
        if x_min == x_max and x_min < 0:
            x_max = 0

        lca_results_plot = bokeh_figure(title=(', '.join([m for m in method])), y_range=list(df.index),
                                        height=BokehPlot.calculate_results_chart_height(
                                            bar_count=df.index.size,
                                            legend_item_count=df.columns.size),
                                        x_range=(x_min, x_max), tools=['hover'],
                                        tooltips=("$name: @$name" if show_legend else "@values"),
                                        sizing_mode="stretch_width", toolbar_location=None)
        lca_results_plot.title.text_font_style = "bold"
        lca_results_plot.title.text_font_size = "12pt"

        if show_legend:
            lca_results_plot.hbar_stack(list(df.columns), height=self.BAR_HEIGHT, y='index', source=column_source,
                                        legend_label=list(df.columns),
                                        fill_color=BokehPlot.get_color_palette(len(df.columns)), line_width=0)
        else:
            lca_results_plot.hbar(y="index", height=self.BAR_HEIGHT, right="values", source=column_source)

        # TODO:
        # Handle scenarios and https://github.com/LCA-ActivityBrowser/activity-browser/issues/622

        if x_min < 0:
            lca_results_plot.x_range.start = x_min

        if x_min > 0 and x_max > 0:
            lca_results_plot.x_range.start = 0
        lca_results_plot.xaxis.axis_label = bw.methods[method].get('unit')

        BokehPlot.style_axis_labels(lca_results_plot.yaxis)

        # Relocate the legend to bottom left to save space
        if show_legend:
            BokehPlot.style_and_place_legend(lca_results_plot, "bottom_left")

        lca_results_plot.ygrid.grid_line_color = None
        lca_results_plot.axis.minor_tick_line_color = None
        lca_results_plot.outline_line_color = None

        self.figure = lca_results_plot

        # Disable context menu as no actions at the moment
        self.view.setContextMenuPolicy(QtCore.Qt.NoContextMenu)

        template = BokehPlot.build_html_bokeh_template()
        html = file_html(lca_results_plot, template=template, resources=None)
        self.page.setHtml(html)
        self.view.setPage(self.page)


class LCAResultsOverview(BokehPlot):
    """" Generate a bar chart comparing the relative LCA scores of the products """

    def on_context_menu(self, *args, **kwargs):
        raise NotImplementedError

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_name = 'LCA Overview'

    def plot(self, df: pd.DataFrame):
        dfp = df.copy()
        dfp.index = dfp['index']
        dfp.drop(dfp.select_dtypes(['object']), axis=1, inplace=True)  # get rid of all non-numeric columns (metadata)
        if "amount" in dfp.columns:
            dfp.drop(["amount"], axis=1, inplace=True)  # Drop the 'amount' col
        if 'Total' in dfp.index:
            dfp.drop("Total", inplace=True)

        # avoid figures getting too large
        dfp.index = [wrap_text(i, max_length=40) for i in dfp.index]
        dfp.columns = [wrap_text_by_separator(i) for i in dfp.columns]

        dfp = dfp.T

        # TODO: What in case of just one reference flow? Dont show this tab?
        dfp = dfp.apply(lambda x: (x / x.max()) if (x.max() > 0) else (x / x.min()), axis=1)  # handle both neg scenario
        dfp = dfp[::-1]

        column_source = ColumnDataSource(dfp)

        # Compute plot height
        plot_height = BokehPlot.calculate_results_chart_height(bar_count=dfp.index.size,
                                                                    legend_item_count=dfp.columns.size)
        min = 0
        if dfp.min().min() < 0:
            min = dfp.min().min()

        lca_results_plot = bokeh_figure(y_range=list(dfp.index), x_range=(min, 1),
                                        height=plot_height, sizing_mode="stretch_width", toolbar_location=None
                                        )

        bar_distribution_value, bar_height = self.get_sub_bar_placement(dfp.columns.size)

        colors = BokehPlot.get_color_palette(len(dfp.columns))
        for column_index in range(0, dfp.columns.size):
            renderer = lca_results_plot.hbar(
                y=dodge('index', mgrid[-bar_distribution_value:bar_distribution_value:dfp.columns.size * 1j][
                    column_index] if dfp.columns.size > 1 else 0.0,
                        range=lca_results_plot.y_range),
                right=dfp.columns[column_index], height=bar_height, source=column_source,
                color=colors[column_index], legend_label=dfp.columns[column_index], name=dfp.columns[column_index])
            hover = HoverTool(tooltips="$name: @$name", renderers=[renderer])
            lca_results_plot.add_tools(hover)

        lca_results_plot.y_range.range_padding = 0.02
        lca_results_plot.ygrid.grid_line_color = None
        BokehPlot.style_axis_labels(lca_results_plot.yaxis)

        lca_results_plot.xaxis.axis_label = "Impact relative to largest"
        lca_results_plot.xaxis.axis_label_text_font_size = "10pt"
        lca_results_plot.xaxis.axis_label_text_font_style = "bold"

        # Relocate the legend to bottom left to save space
        BokehPlot.style_and_place_legend(lca_results_plot, "bottom_left")

        self.figure = lca_results_plot

        # Disable context menu as no actions at the moment
        self.view.setContextMenuPolicy(QtCore.Qt.NoContextMenu)

        template = BokehPlot.build_html_bokeh_template()
        html = file_html(lca_results_plot, template=template, resources=None)
        self.page.setHtml(html)
        self.view.setPage(self.page)

    def get_sub_bar_placement(self, column_size):
        """
        Computes the relative distances within the yaxis for the placement of sub-bars in the area allocated for a bar
        """
        bar_distribution_value = 0.0
        bar_height = 0.4
        if column_size == 2:
            bar_distribution_value = 0.1
            bar_height = 0.2
        if column_size >= 3:
            bar_distribution_value = 0.3
            bar_height = 0.15
        if column_size > 6:
            bar_distribution_value = 0.4
            bar_height = 0.08

        return bar_distribution_value, bar_height


class HeatMapPlot(BokehPlot):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_name = 'Overview'
        self.data: pd.DataFrame = None
        self.context_menu_actions: list = None
        self.chart_bridge: ChartBridge = None
        self.channel: QtWebChannel.QWebChannel = None
        self.hover_callback = None

    # The colour mapping methods are inspired from the blog post of Ben Southgate (https://bsouthga.dev/posts/color-
    # gradients-with-python)
    def to_hex(self, colour: int) ->str:
        assert colour >= 0, f"Converting an integer to a hexadecimal colour code requires a value: 0 >= value <=255. Received: {colour}"
        assert colour <= 255, f"Converting an integer to a hexadecimal colour code requires a value: 0 >= value <=255. Received: {colour}"
        return "0{0:x}".format(colour) if colour < 16 else "{0:x}".format(colour)

    def to_rgb(self,colour: hex):
        return [int(colour[i:i+2], 16) for i in range(1, 6, 2)]

    def list_of_colours(self, gradient):
        return ["#"+"".join([self.to_hex(rgb[j]) for j in range(3)]) for rgb in gradient]

    def linear_colour_gradient(self, start_hex: hex, finish_hex: hex='#FFFFFF', n = 100):
        """ Creates a linear colour graident given a starting and final colour to define a range and the number of steps.
        """
        start = self.to_rgb(start_hex)
        finish = self.to_rgb(finish_hex)
        # start of the list
        rgb_list = [start]
        for color in range(1,n):
            color_vector = [
                int(start[j] + (float(color)/(n-1))*(finish[j]-start[j])) for j in range(3)
            ]
            rgb_list.append(color_vector)
        return self.list_of_colours(rgb_list)

    def polylinear_colour_gradient(self, colours, n):
        """ Creates a colour gradient from several different colour gradients"""
        n_per_gradient = int(float(n) / (len(colours) -1))

        gradient = self.linear_colour_gradient(colours[0], colours[1], n_per_gradient)

        if len(colours) > 1:
            for col in range(1, len(colours)-1):
                next_gradient = self.linear_colour_gradient(colours[col], colours[col+1], n_per_gradient)
                gradient += next_gradient
        return gradient

    def calculate_chart_height(self, bar_count):
        return 300 + (200 * bar_count)

    def plot(self, df: pd.DataFrame, context_menu_actions: [] = None, x_axis = None, y_axis=None):
        self.context_menu_actions = context_menu_actions
        if x_axis is None:
            x_axis = 'variable'
        if y_axis is None:
            y_axis = 'index'
        # data preparation steps
        self.data = df.copy()
        self.data.index = self.data['index']
        # create a "table" that can allow us to create the proportions for plotting
        self.proportionate_data = self.data.copy()
        # involving removing those datatypes we need to avoid for division
        # including the 'Total' and 'amount' columns
        if "amount" in self.proportionate_data.columns:
            self.proportionate_data.drop(["amount"], axis=1, inplace=True)  # Drop the 'amount' col
        if 'Total' in self.proportionate_data.index:
            self.proportionate_data.drop("Total", inplace=True)
        self.proportionate_data.drop(self.proportionate_data.select_dtypes(['object']), axis=1, inplace=True)
        self.proportionate_data = self.proportionate_data.divide(self.proportionate_data.max(axis=0)).multiply(100)
        # then transform the proportion data into a usable form and reset the indices
        self.proportionate_data = self.proportionate_data.melt(ignore_index=False)
        self.proportionate_data.index = pd.MultiIndex.from_arrays([self.proportionate_data.index.tolist(),
                                    self.proportionate_data.loc[:, 'variable'].tolist()])
        self.proportionate_data.rename({'value': 'relative'}, axis=1, inplace=True)

        # now we need to arrange the dataset with the physical (non-transformed) data
        self.data.drop(['name', 'location', 'database'], axis=1, inplace=True)
        self.data = self.data.melt(id_vars=['unit', 'amount'], value_vars=self.proportionate_data['variable'].unique(),
                                   ignore_index=False)
        self.data['index'] = self.data.index
        self.data.index = pd.MultiIndex.from_arrays([self.data.index.tolist(),
                                                                   self.data.loc[:, 'variable'].tolist()])
        self.data.rename({'value': 'real'}, axis=1, inplace=True)

        #Structure the dataframe so that the x and y axis are factors in columns
        self.data = pd.concat([self.data, self.proportionate_data.loc[:, 'relative']], axis=1)
        self.data['relative'] = self.data['relative'].astype(int)

        width = self.size.width()/2
        height = self.size.height()/1.5

        self.data.loc[:, x_axis] = BokehPlot.reformat_labels(list(self.data.loc[:, x_axis]))
        x_labels = list(self.data.loc[:, x_axis].drop_duplicates())
        self.data.loc[:, y_axis] = BokehPlot.reformat_labels(list(self.data.loc[:, y_axis]))
        y_labels = list(self.data.loc[:, y_axis].drop_duplicates())


        # set the colours and the tooltips for the figure
        colors = self.polylinear_colour_gradient(["#4646C3", "#56D7C3", "#E36747"], 100)
        _tooltips_ = HoverTool(tooltips=[
            ("Score:", "@real"),
            ("X:", f"@{x_axis}"),
            ("Y:", f"@{y_axis}"),
            ("per ", "@amount @unit")],
            attachment="right",
            anchor='bottom_right'
        )
        heatmap = bokeh_figure(
            title="Overall results",
            x_range=x_labels,
            y_range=y_labels,
            tools="box_zoom", toolbar_location='above',
            width=int(width), height=int(height)
        )
        heatmap.add_tools(_tooltips_)
        heatmap.grid.grid_line_color=None
        heatmap.axis.axis_line_color=None
        #heatmap.major_tick_line_color=None
        heatmap.axis.major_label_standoff=2
        heatmap.axis.major_label_text_font_size="8px"
        heatmap.yaxis.major_label_orientation=math.pi/3
        heatmap.xaxis.major_label_orientation=math.pi/3
        f = heatmap.rect(x=x_axis,
                     y=y_axis,
                     width=1,
                     height=1,
                     source=self.data,
                     fill_color=linear_cmap('relative', colors, low=0, high=100),
                     line_color=None,
                     )

        # adding a key
        heatmap.add_layout(f.construct_color_bar(
            major_label_text_font_size="9px",
            ticker=BasicTicker(desired_num_ticks=3),
            formatter=PrintfTickFormatter(format="%d%%"),
            label_standoff=6,
            border_line_color=None,
            padding=5,
        ), 'right')
        #column_layout = bokeh.layouts.column([heatmap])
        #column_layout.sizing_mode='scale_width'
        # the page for presenting the plot
        template = BokehPlot.build_html_bokeh_template(
            add_context_menu_communication=self.context_menu_actions is not None,
            disable_horizontal_scroll=False)
        html = file_html(heatmap, template=template, resources=None)
        self.page.setHtml(html)
        self.view.setPage(self.page)

class ContributionPlot(BokehPlot):
    BAR_HEIGHT = 0.6

    def __init__(self):
        super().__init__()
        self.plot_name = 'Contributions'
        self.plot_data: pd.DataFrame = None
        self.context_menu_actions: list = None
        self.chart_bridge: ChartBridge = None
        self.channel: QtWebChannel.QWebChannel = None
        self.hover_callback = None

    def plot(self, df: pd.DataFrame, unit: str = None, context_menu_actions: [] = None,
             is_relative: bool = True):
        """ Plot a horizontal stacked bar chart for the process and elementary flow contributions. """
        self.context_menu_actions = context_menu_actions

        # Copy, clean and transform the dataframe for plotting
        self.plot_data = df.copy()
        self.plot_data.index = self.plot_data['index']
        self.plot_data.drop(self.plot_data.select_dtypes(['object']), axis=1,
                            inplace=True)  # Remove all non-numeric columns (metadata)
        if 'Total' in self.plot_data.index:
            totals = list(self.plot_data.loc["Total"])
            totals.reverse()
            self.plot_data.drop("Total", inplace=True)
        self.plot_data = self.plot_data.fillna(0)
        self.plot_data = self.plot_data.T
        self.plot_data = self.plot_data[::-1]  # Reverse sort the data as bokeh reverses the plotting order

        # Avoid figures getting too large horizontally by text wrapping
        self.plot_data.index = pd.Index([wrap_text(str(i), max_length=40) for i in self.plot_data.index])
        self.plot_data.columns = pd.Index([wrap_text(str(i), max_length=40) for i in self.plot_data.columns])

        # Handle negative values
        has_negative_values = (self.plot_data.values < 0).any()
        has_positive_values = (self.plot_data.values > 0).any()
#        positive_df = self.plot_data.copy()
#        if has_negative_values:
#            negative_df = self.plot_data[self.plot_data < 0]
#            negative_df = negative_df.fillna(0)
#            positive_df = self.plot_data[self.plot_data > 0]
#            positive_df = positive_df.fillna(0)

        # Compute plot height # TODO this calculation seems wrong, it produced 296 pixels for a plot height
        plot_height = BokehPlot.calculate_bar_chart_height(bar_count=self.plot_data.index.size,
                                                                legend_item_count=self.plot_data.columns.size)

#        # Prepare the plot and add stacked bars
#        contribution_plot = bokeh_figure(y_range=list(self.plot_data.index), toolbar_location=None,
#                                         height=plot_height,
#                                         sizing_mode="stretch_width")
#
        if has_positive_values:
            positive = ColumnDataSource(self.plot_data)
            contribution_plot = bokeh_figure(y_range=positive.data['index'].tolist(), toolbar_location=None,
                                             height=plot_height,
                                             sizing_mode="stretch_width"
                                             )

            contribution_plot.hbar_stack(stackers=positive.column_names[1:], height=self.BAR_HEIGHT, y='index',
                                         source=positive,
                                         legend_label=positive.column_names[1:],
                                         fill_color=BokehPlot.get_color_palette(len(positive.column_names)-1),
                                         line_width=0)

        if has_negative_values:
            negative = ColumnDataSource(self.plot_data)
            contribution_plot = bokeh_figure(y_range=negative.data['index'].tolist(), toolbar_location=None,
                                             height=plot_height,
                                             sizing_mode="stretch_width"
                                             )

            contribution_plot.hbar_stack(stackers=negative.column_names[1:], height=self.BAR_HEIGHT, y='index',
                                         source=negative,
                                         legend_label=negative.column_names[1:],
                                         fill_color=BokehPlot.get_color_palette(len(negative.column_names)-1, True),
                                         line_width=0)
            source_totals = ColumnDataSource(
                data=dict(base=list(self.plot_data.index), lower=np.zeros(self.plot_data.index.size), upper=totals))
            w = Whisker(source=source_totals, base="base", upper="upper", lower="lower", dimension="width",
                        level="overlay", line_color="red", line_width=1.5) # TODO: Dont add wisker for positive bar
            w.upper_head.line_width = 1.5
            w.lower_head.line_width = 1.5
            w.upper_head.line_color = 'red'
            w.lower_head.line_color = 'red'
            contribution_plot.add_layout(w)

            total_legend = Label(x=0, y=-45, x_units='screen', y_units='screen',
                                 text=' H Aggregate value ', text_color='red', border_line_color='black',
                                 border_line_alpha=1.0, text_font_size="10pt", text_font_style="bold",
                                 background_fill_color='white', background_fill_alpha=1.0)
            contribution_plot.add_layout(total_legend)

#        if not has_negative_values:
#            contribution_plot.x_range.start = 0 # TODO find an alternative approach, this is a read-only property

        if unit:
            contribution_plot.xaxis.axis_label = unit
            contribution_plot.xaxis.axis_label_text_font_size = "10pt"
            contribution_plot.xaxis.axis_label_text_font_style = "bold"

        # Relocate the legend to bottom left to save space
        BokehPlot.style_and_place_legend(contribution_plot, (-100, 0))

        # Handle styling
        contribution_plot.ygrid.grid_line_color = None
        contribution_plot.axis.minor_tick_line_color = None
        contribution_plot.outline_line_color = None
        BokehPlot.style_axis_labels(contribution_plot.yaxis)

        self.figure = contribution_plot

        # Handle context menu:
        self.init_context_menu()

        # Add tooltip on hover
        hover_tool_plot = HoverTool(callback=self.hover_callback, tooltips="$name: @$name")
        contribution_plot.add_tools(hover_tool_plot)

        # Create static HTML and render in web-view (this can be exported - will contain hover interaction)
        template = BokehPlot.build_html_bokeh_template(
            add_context_menu_communication=self.context_menu_actions is not None,
            disable_horizontal_scroll=True)
        html = file_html(contribution_plot, template=template, resources=None)
        self.page.setHtml(html)
        self.view.setPage(self.page)

    def init_context_menu(self):
        # Prepare context menu actions Array[tuples(label, function callback)]
        if self.context_menu_actions is not None:
            self.chart_bridge = ChartBridge(self)
            self.channel = QtWebChannel.QWebChannel()
            self.channel.registerObject('chartBridge', self.chart_bridge)
            self.page.setWebChannel(self.channel)
            self.view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.view.customContextMenuRequested.connect(self.on_context_menu)
            self.hover_callback = CustomJS(code="""
                                                            //console.log(cb_data)
                                                            window.lastHover = {}
                                                            window.lastHover.x = cb_data.geometry.x
                                                            window.lastHover.y = cb_data.geometry.y
                                                            """)
        else:
            self.hover_callback = None
            self.view.setContextMenuPolicy(QtCore.Qt.NoContextMenu)

    def on_context_menu(self, pos):
        """
        Finds the bar and sub-bar, if position of right-click is correct, prepares context menu with actions passed
        and shows it with the co-ordinates passed to ChartBridge object
        @param pos: Position of right-click within application window
        """
        if not self.context_menu_actions or self.chart_bridge.context_menu_x is None or self.chart_bridge.context_menu_y is None:
            return

        bar_margin = 1 - self.BAR_HEIGHT
        bar_index = math.floor(self.chart_bridge.context_menu_y)
        bar_index_start = bar_index + (bar_margin / 2)
        bar_index_end = bar_index + 1 - (bar_margin / 2)
        if (self.chart_bridge.context_menu_y > 0 and bar_index < self.plot_data.index.size
                and bar_index_start <= self.chart_bridge.context_menu_y <= bar_index_end):  # self.chart_bridge.context_menu_x > 0 and

            data_table = self.plot_data.copy()
            is_sub_bar_negative = self.chart_bridge.context_menu_x < 0

            if is_sub_bar_negative:
                data_table = data_table[data_table < 0]
                data_table = data_table.abs()
            else:
                data_table = data_table[data_table > 0]

            data_table = data_table.fillna(0)

            prev_val = 0
            for col_index, column in enumerate(list(data_table.columns), start=0):
                if data_table.iloc[bar_index][column] == 0:
                    continue

                prev_val = data_table.iloc[bar_index][column] + prev_val
                if abs(self.chart_bridge.context_menu_x) < prev_val:
                    # bar_label = self.data.index[bar_index]
                    # sub_bar_label = column

                    context = QMenu(self)
                    for action_name, _action in self.context_menu_actions:
                        context_menu_item = QAction(action_name, self)
                        context_menu_item.triggered.connect(
                            lambda: _action(bar_index=bar_index, sub_bar_index=col_index))
                        context.addAction(context_menu_item)
                    context.popup(self.mapToGlobal(pos))
                    break


class ChartBridge(QObject):
    """
    Chart bridge is used to communicate the co-ordinates in the chart where the user performs a right-click
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.context_menu_x = 0
        self.context_menu_y = 0

    @Slot(str, name="set_context_menu_coordinates")
    def set_context_menu_coordinates(self, args: str):
        """
        The set_context_menu_coordinates is called from the JS with the co-ordinates and these co-ordinates are then used to
    open the context menu from python.
        Args:
            args: string of a serialized json dictionary describing
            - x: X axis co-ordinate (Index of the sub-bar(part-of bar) on which context menu was opened)
            - y: Y axis co-ordinate (Index of the bar on which context menu was opened)
        """
        data_dict = json.loads(args)
        self.context_menu_x = data_dict['x']
        self.context_menu_y = data_dict['y']


class MonteCarloPlot(BokehPlot):
    """ Monte Carlo plot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_name = 'Monte Carlo'

    def on_context_menu(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self, df: pd.DataFrame, method: tuple):
        p = bokeh_figure(tools=['wheel_zoom', 'pan'], background_fill_color="#fafafa", toolbar_location=None,
                         sizing_mode="stretch_width",
                         )
        p.toolbar.active_scroll = p.select_one(WheelZoomTool)
        colors = BokehPlot.get_color_palette(df.columns.size)
        i = 0
        for col in df.columns:
            hist, edges = np.histogram(df[col], density=False)
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=colors[i], line_width=0,
                   alpha=0.5, legend_label=col)
            span = Span(location=df[col].mean(), dimension='height', line_color=colors[i], line_width=2)
            p.renderers.append(span)
            i = i + 1

        # Relocate the legend to bottom left to save space
        BokehPlot.style_and_place_legend(p, "center")
        p.xaxis.axis_label = bw.methods[method]["unit"]
        p.yaxis.axis_label = 'Count'
        p.y_range.start = 0
        p.y_range.bounds = (0, None)

        self.figure = p

        # Disable context menu as no actions at the moment
        self.view.setContextMenuPolicy(QtCore.Qt.NoContextMenu)

        template = BokehPlot.build_html_bokeh_template()
        html = file_html(p, template=template, resources=None)
        self.page.setHtml(html)
        self.view.setPage(self.page)
