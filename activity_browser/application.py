# -*- coding: utf-8 -*-
import os
from .controllers import controllers
from .layouts import MainWindow


class Application(object):
    CURRENT_PATH = os.path.dirname(__file__)
    CSS_STYLE = os.path.join(CURRENT_PATH, 'static', 'css', 'main.css')
    def __init__(self):
        self.main_window = MainWindow()

        # Instantiate all the controllers.
        # -> Ensure all controller instances have access to the MainWindow
        # object, this propagates the 'AB' icon to all controller-handled
        # dialogs and wizards.
        for attr, controller in controllers.items():
            setattr(self, attr, controller(self.main_window))
        with open(Application.CSS_STYLE, "r") as fh:
            self.main_window.setStyleSheet(fh.read())

    def show(self):
        self.main_window.showMaximized()

    def close(self):
        self.main_window.close()

    def deleteLater(self):
        self.main_window.deleteLater()
