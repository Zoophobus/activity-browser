
from PySide2 import QtWidgets


class ABResultsCheckboxes(QtWidgets.QFrame):
    """A customised arrangement for viewing and selecting checkboxes

    """

    class ABCheckBox(QtWidgets.QCheckBox):
        def __init__(self, label: str = None, index: int = -1):
            self.index = index
            self.original_data = label
            if not isinstance(label, str):
                label = str(label)
            super().__init__(label)

        @property
        def at(self) -> int:
            return self.index

        @at.setter
        def at(self, index: int) -> None:
            self.index = index

        @property
        def button(self) -> object:
            return self.original_data

        @button.setter
        def button(self, label: object) -> None:
            self.original_data = label

    def __init__(self, header: QtWidgets.QVBoxLayout = None):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(header)
        self.button_group = QtWidgets.QButtonGroup()
        self.button_group.setExclusive(False)
        self.boxes = list()
        self.setLayout(self.layout)

    def update_boxes(self, content: list()) -> None:
        self.boxes.clear()
        for i, lbl in enumerate(content):
            bttn = ABResultsCheckboxes.ABCheckBox(lbl, i)
            self.boxes.append(bttn)
            self.button_group.addButton(bttn)
            self.layout.addWidget(bttn)

        self.layout.addStretch(1)
        self.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.setLayout(self.layout)
