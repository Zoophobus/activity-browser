from PySide2 import QtGui, QtCore, QtWidgets
from typing import Optional, Union

from activity_browser.ui.tables.views import ABDataFrameView
from .delegates import CheckboxDelegate
from activity_browser.signals import signals
from .models.lca_filter_model import FilterReferencesModel, FilterMethodsModel, FilterScenariosModel

class FilterReferencesTable(ABDataFrameView):
    """
    Displays some of the data for the reference flows in the LCA.

    Contains a Checkbox delegate for controlling the presentation of the contents.

    Does not contain numeric demand scalars/vectors and is otherwise read-only.
    """

    def __init__(self, parent = None):
        super().__init__(parent)
        self.verticalHeader().setVisible(False)
        self.setItemDelegateForColumn(3, CheckboxDelegate(self))
        self.setSizePolicy(QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum
        ))
        self.setToolTip("Check the boxes to set the visibility of the Reference flows (Remember to press the Update button)")
        self.model = FilterReferencesModel(parent=parent)
        self.model.updated.connect(self.update_proxy_model)
        self.model.updated.connect(self.custom_view_sizing)
        self.model.sync()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """
        Defines the use of a single mouse click within the table.

        """
        idx = None
        if event.button() == QtCore.Qt.LeftButton:
            proxy = self.indexAt(event.pos())
            if proxy.column() == 3:
                idx = proxy.row()
                self.model.toggle(idx)
                self.model.sync()
                signals.lca_results_filter.emit(idx, "Reference Flows")

class FilterMethodsTable(ABDataFrameView):
    """
    Displays the name and a checkbox for the Methods in the LCA.

    Toggling the checkbox will change the visibility of the Methods in the LCA results.

    Does not contain numeric scalars/vectors.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.verticalHeader().setVisible(False)
        self.setItemDelegateForColumn(1, CheckboxDelegate(self))
        self.setSizePolicy(QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum
        ))
        self.setToolTip("Check the boxes to set the visibility of the assessment methods (Remember to press the Update button)")
        self.model = FilterMethodsModel(parent=parent)
        self.model.updated.connect(self.update_proxy_model)
        self.model.updated.connect(self.custom_view_sizing)
        self.model.sync()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """
        Defines the use of a single mouse click within the table.

        """
        if event.button() == QtCore.Qt.LeftButton:
            proxy = self.indexAt(event.pos())
            if proxy.column() == 1:
                idx = proxy.row()
                self.model.toggle(idx)
                self.model.sync()
                signals.lca_results_filter.emit(idx, "Impact Assessment Methods")


class FilterScenariosTable(ABDataFrameView):
    """
    Displays the name and a checkbox for the Scenarios (if present) in the LCA.

    Toggling the checkbox will change the visibility of the respective scenarios in the LCA results.

    Does not contain numeric scalars/vectors
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.verticalHeader().setVisible(False)
        self.setItemDelegateForColumn(1, CheckboxDelegate(self))
        self.setSizePolicy(QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum
        ))
        self.setToolTip("Check the boxes to set the visibility of the scenarios (Remember to press the Update button)")
        self.model = FilterScenariosModel(parent=parent)
        self.model.updated.connect(self.update_proxy_model)
        self.model.updated.connect(self.custom_view_sizing)
        self.model.sync()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """
        Defines the use of a single mouse click within the table.

        """
        if event.button() == QtCore.Qt.LeftButton:
            proxy = self.indexAt(event.pos())
            if proxy.column() == 1:
                idx = proxy.row()
                self.model.toggle(idx)
                self.model.sync()
                signals.lca_results_filter.emit(idx, "Scenarios")