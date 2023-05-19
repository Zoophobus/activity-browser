import pandas

import brightway2 as bw
from bw2data.backends.peewee import ActivityDataset

from .base import PandasModel
from activity_browser.bwutils import commontasks as bc

# TODO Move the creation of the dataframe into the __init__ method
class FilterReferencesModel(PandasModel):
    HEADERS = [
        "Amount", "Unit", "Product", "Activity", "Location", "Database", "Set visible"
    ]
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        fus = bw.calculation_setups.get(self.parent.parent.cs_name, {}).get('inv', [])# TODO replace self.current_cs with an attribute providing the name of the current calculation setup
        self._dataframe = pandas.DataFrame([
            self.build_row(key, amount) for func_unit in fus
            for key, amount in func_unit.items()
        ], columns=self.HEADERS[:-1])
        self._dataframe.insert(6, self.HEADERS[-1], [True for i in range(self._dataframe.shape[0])], allow_duplicates=True)
        self._dataframe.columns = self.HEADERS

    def toggle(self, idx_y: int):
        self._dataframe.iloc[idx_y, 6] = not self._dataframe.iloc[idx_y, 6]

    def toggled(self):
        return self._dataframe.iloc[:, 6]

    def sync(self):
        self.updated.emit()

    def build_row(self, key: tuple, amount: float = 1.0) -> dict:
        try:
            act = bw.get_activity(key)
            if act.get("type", "process") != "process":
                raise TypeError("Activity is not of type 'process'")
            row = {
                key: act.get(bc.AB_names_to_bw_keys[key], "")
                for key in self.HEADERS[:-1]
            }
            row.update({"Amount": amount, "key": key})
            return row
        except (TypeError, ActivityDataset.DoesNotExist):
            print("Could not load key in Calculation Setup: ", key)
            return {}


class FilterMethodsModel(PandasModel):
    HEADERS = ["Name", "Unit", "# CFs", "method", "Set visible"]
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self._dataframe = pandas.DataFrame([
            self.build_row(method)
            for method in bw.calculation_setups[self.parent.parent.cs_name].get("ia", [])
        ], columns=self.HEADERS[:-1])
        self._dataframe.insert(4, self.HEADERS[-1], [True for i in range(self._dataframe.shape[0])], allow_duplicates=True)
        self._dataframe.columns = self.HEADERS

    def toggle(self, idx_y: int):
        self._dataframe.iloc[idx_y, 4] = not self._dataframe.iloc[idx_y, 4]

    def toggled(self):
        return self._dataframe.iloc[:, 4]

    def sync(self):
        self.updated.emit()

    def build_row(self, method: tuple) -> dict:
        method_metadata = bw.methods[method]
        return {
            "Name": ', '.join(method),
            "Unit": method_metadata.get('unit', "Unknown"),
            "# CFs": method_metadata.get('num_cfs', 0),
            "method": method,
        }


class FilterScenariosModel(PandasModel):
    HEADERS = ["Name", "Set visible"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self._dataframe = self.parent.parent.mlca.scenario_dataframe[['name']]
        self._dataframe.insert(1,self.HEADERS[-1], [True for i in range(self._dataframe.shape[0])], allow_duplicates=True)
        self._dataframe.columns = self.HEADERS

    def toggle(self, idx_y: int):
        self._dataframe.iloc[idx_y, 1] = not self._dataframe.iloc[idx_y, 1]

    def toggled(self):
        return self._dataframe.iloc[:, 1]

    def sync(self):
        self.updated.emit()