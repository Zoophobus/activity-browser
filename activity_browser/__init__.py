# -*- coding: utf-8 -*-
import os
import sys
import traceback

from PySide2.QtCore import QSysInfo, __version__ as qt_version
from PySide2.QtWidgets import QApplication

import chromedriver_binary  # required to add chrome driver path in environment PATH variable

from .application import Application
from .info import __version__

# https://bugreports.qt.io/browse/QTBUG-87014
# https://bugreports.qt.io/browse/QTBUG-85546
# https://github.com/mapeditor/tiled/issues/2845
# https://doc.qt.io/qt-5/qoperatingsystemversion.html#MacOSBigSur-var
if QSysInfo.productType() == "osx" and (
        QSysInfo.productVersion() == "10.16" or QSysInfo.productVersion() == "11.0"
):
    os.environ["QT_MAC_WANTS_LAYER"] = "1"
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
    print("Warning! The currently used version of Qt cannot properly handle BigSur yet.")


if QSysInfo.productType() in ["arch","nixos"]:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--no-sandbox"
    print("Info: QtWebEngine sandbox disabled")

def run_activity_browser():
    qapp = QApplication(sys.argv)

    # qapp.setFont(default_font)
#    qapp.setStyleSheet(
#        '''
#            * { background-color: #f0f0f0; color #101010}
#            QWidget>QWidget>QWidget>QWidget>QWidget>QWidget { background-color: #FFFFFF; color: #101010 }
#            QWidget>QTableView { background-color: #FFFFFF; color: #101010 }
#            QWidget>QTableWidget { background-color: #FFFFFF; color: #101010 }
#        '''
#    )
    application = Application()
    application.show()
    print("Qt Version:", qt_version)

    def exception_hook(*args):
        print(''.join(traceback.format_exception(*args)))

    sys.excepthook = exception_hook

    sys.exit(qapp.exec_())
