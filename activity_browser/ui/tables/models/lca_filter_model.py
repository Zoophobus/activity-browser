import pandas

from .base import PandasModel
from activity_browser.signals import signals


class FilterReferencesModel(PandasModel):
    HEADERS = ["Name", "Key", "Database", "Show/Hide"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self._dataframe = self.parent.parent.mlca.reference_dataframe[["reference_name", "reference_key"]]
        tmp_df = pandas.DataFrame(self._dataframe['reference_key'].tolist(), index=self._dataframe.index)
        self._dataframe = self._dataframe.assign(Database=tmp_df.iloc[:, 0], Key=tmp_df.iloc[:, 1])
        self._dataframe.drop("reference_key", axis=1, inplace=True)
        self._dataframe.insert(3, self.HEADERS[-1], [True for i in range(self._dataframe.shape[0])], allow_duplicates=True)
        self._dataframe.columns = self.HEADERS

    def toggle(self, idx_y: int, check: bool = None):
        if check is not None:
            if self._dataframe.iloc[idx_y, 3] == check:
                return 0
            self._dataframe.iloc[idx_y, 3] = not self._dataframe.iloc[idx_y, 3]
        else:
            self._dataframe.iloc[idx_y, 3] = not self._dataframe.iloc[idx_y, 3]
            signals.all_reference_flows_checked.emit(self._dataframe.iloc[:, 3].all())
        return 1

    def sync(self):
        self.updated.emit()


class FilterMethodsModel(PandasModel):
    HEADERS = ["Name", "Show/Hide"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self._dataframe = self.parent.parent.mlca.methods_dataframe[['label']]
        self._dataframe.insert(1, self.HEADERS[-1], [True for i in range(self._dataframe.shape[0])], allow_duplicates=True)
        self._dataframe.columns = self.HEADERS

    def toggle(self, idx_y: int, check: bool = None):
        if check is not None:
            if self._dataframe.iloc[idx_y, 1] == check:
                return 0
            self._dataframe.iloc[idx_y, 1] = not self._dataframe.iloc[idx_y, 1]
        else:
            self._dataframe.iloc[idx_y, 1] = not self._dataframe.iloc[idx_y, 1]
            signals.all_impact_categories_checked.emit(self._dataframe.iloc[:, 1].all())
        return 1

    def sync(self):
        self.updated.emit()


class FilterScenariosModel(PandasModel):
    HEADERS = ["Name", "Show/Hide"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self._dataframe = self.parent.parent.mlca.scenario_dataframe[['name']]
        self._dataframe.insert(1,self.HEADERS[-1], [True for i in range(self._dataframe.shape[0])], allow_duplicates=True)
        self._dataframe.columns = self.HEADERS

    def toggle(self, idx_y: int, check: bool = None):
        if check is not None:
            if self._dataframe.iloc[idx_y, 1] == check:
                return 0
            self._dataframe.iloc[idx_y, 1] = not self._dataframe.iloc[idx_y, 1]
        else:
            self._dataframe.iloc[idx_y, 1] = not self._dataframe.iloc[idx_y, 1]
            signals.all_scenarios_checked.emit(self._dataframe.iloc[:, 1].all())
        return 1

    def sync(self):
        # and synchronize the all checkbox too
        self.updated.emit()
