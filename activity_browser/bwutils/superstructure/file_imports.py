from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
import ast
from PySide2.QtWidgets import QMessageBox, QFileDialog

from typing import Optional, Union


from ..metadata import AB_metadata
from ..errors import (
    ImportCanceledError, ActivityProductionValueError, IncompatibleDatabaseNamingError,
    InvalidSDFEntryValue, ExchangeErrorValues, ABError
)


class ABPopup(QMessageBox):
    """
    Holds AB defined message boxes to enable a more consistent popup message structure
    """
    def __init__(self, parent=None, title: str = None):
        super().__init__(parent)
        self.data_frame = None
        self.message = None

    def dataframe(self, data: pd.DataFrame, columns: list = None):
        self.data_frame = data
        self.data_frame = self.data_frame.loc[:, columns]
        self.data_frame.index = self.data_frame.index.astype(str)
#        for column in columns:
#            self.data_frame[column] = pd.Series(self.data_frame[column].values.flatten())

    def dataframe_to_str(self, double: set = {'to key', 'from key'} ):

        #define a function to write the lines
        separator = lambda output, _type='': str('\t'*2) + output if _type in double else (
            str(' '*8) + output if len(output) < 20 else '\t' + output[0:19] + '...'
        )
        #Writes out the header line
        conversion = 'index'
        for column in range(0, len(self.data_frame.columns)):
            conversion = conversion + separator(self.data_frame.columns[column], self.data_frame.columns[column])
        conversion = conversion + '\n'
        #writes out the table body
        for row in self.data_frame.index:
            conversion = conversion + row
            for column in self.data_frame.columns:
                conversion = conversion + separator(str(self.data_frame.loc[row, column]))
            conversion = conversion + '\n'
        return conversion

    def abQuestion(self, title, message, button1, button2) -> QMessageBox:
        self.setWindowTitle(title)
        self.setText(message)
        self.setIcon(QMessageBox.Question)
        self.setStandardButtons(button1 | button2)
        self.setDefaultButton(button1)
        if self.data_frame is not None:
            self.setDetailedText(self.dataframe_to_str())
        return self.exec_()


    def abWarning(self, title, message, button1, button2) -> QMessageBox:
        self.setWindowTitle(title)
        self.setText(message)
        self.setIcon(QMessageBox.Warning)
        self.setStandardButtons(button1 | button2)
        self.setDefaultButton(button1)
        if self.data_frame is not None:
            self.setDetailedText(self.dataframe_to_str())
        return self.exec_()

    def abCritical(self, title, message, button1, button2) -> QMessageBox:
        self.setWindowTitle(title)
        self.setText(message)
        self.setIcon(QMessageBox.Critical)
        self.setStandardButtons(button1 | button2)
        self.setDefaultButton(button1)
        if self.data_frame is not None:
            self.setDetailedText(self.dataframe_to_str())
        return self.exec_()

    def save_dataframe(self, dataframe: pd.DataFrame) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            parent=self, caption="Choose the location to save the dataframe",
            filter="All Files (*.*);; CSV (*.csv);; Excel (*.xlsx)",
        )
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            dataframe.to_excel(filepath, index=False)
        else:
            dataframe.to_csv(filepath, index=False)


class ABFileImporter(ABC):
    """
    Activity Browser abstract base class for scenario file imports

    Contains a set of static methods for checking the file contents
    to conform to the desired standard. These include:
    - correct spelling of key and database names (checking they match)
    - correct spelling of databases (if few instances are found)
    - that all production exchanges do not have a value of 0
    - that NAs are properly interpreted
    """
    ABStandardProcessColumns = {'from activity name', 'from reference product', 'to reference product', 'to location',
                                'from location', 'to activity name', 'from key', 'flow type', 'from database',
                                'to database', 'to key', 'from unit', 'to unit'}

    ABScenarioColumnsErrorIfNA = {'from key', 'flow type', 'to key'}
    ABStandardBiosphereColumns = {'from categories', 'to categories'}



    def __init__(self):
        pass

    @abstractmethod
    def read_file(self, path: Optional[Union[str, Path]], **kwargs):
        """Abstract method must be implemented in child classes."""
        return NotImplemented

    @staticmethod
    def database_and_key_check(data: pd.DataFrame) -> None:
        """Will check the values in the 'xxxx database' and the 'xxxx key' fields.
        If the database names are incongruent an IncompatibleDatabaseNamingError is raised.
        The source and destination keys are provided for the first exchange where
        this error occurs.
        """
        try:
            for ds in zip(data['from database'], data['from key'], data['to database'], data['to key'], data['from activity name'], data['to activity name']):
                if ds[0] != ds[1].split(',')[0][2:-1] or ds[2] != ds[3].split(',')[0][2:-1]:
                    msg = "Error in importing file with activity {} and {}".format(ds[4], ds[5])
                    raise IncompatibleDatabaseNamingError()
        except IncompatibleDatabaseNamingError as e:
            print(msg)
            raise e

    @staticmethod
    def production_process_check(data: pd.DataFrame, scenario_names: list) -> None:
        """ Runs a check on a dataframe over the scenario names (provided by the second argument)
        If for a production exchange a value of 0 is observed for one of the scenarios an
        ActivityProductionValueError is thrown with the source and destination activity names of the
        exchanges being provided
        """
        failed = pd.DataFrame({})
        try:
            for scenario in scenario_names:
                failed = pd.concat([data.loc[(data.loc[:, 'flow type'] == 'production') & (data.loc[:, scenario] == 0.0)], failed])
            if not failed.empty:
                msg = "Error with the production value in the exchange between activity {} and {}".format(failed['from activity name'], failed['to activity name'])
                raise ActivityProductionValueError()
        except ActivityProductionValueError as e:
            print(msg)
            raise e

    @staticmethod
    def na_value_check(data: pd.DataFrame, fields: list) -> None:
        """ Runs checks on the dataframe to ensure that those fields specified by the field argument do not
        contain NaNs.
        If an NaN is discovered an InvalidSDFEntryValue Error is thrown that contains two lists:
        The first contains the list of the source activity names, the second the destination activity names
        of the exchange
        """
        hasNA = pd.DataFrame({})
        try:
            for field in fields:
                hasNA = pd.concat([data.loc[data[field].isna()], hasNA])
            if not hasNA.empty:
                msg = "Error with NA's in the exchange between activity {} and {}".format(hasNA['from activity name'], hasNA['to activity name'])
                raise InvalidSDFEntryValue()
        except InvalidSDFEntryValue as e:
            print(msg)
            raise e

    @staticmethod
    def check_for_calculation_errors(data: pd.DataFrame) -> None:
        """
        Will check for calculation errors in the scenario exchanges columns indicate the first elements in the
        scenario difference file that contain an ERROR value (only deals with divide by zero and NaN manipulations).
        """
        scen_cols = set(data.columns).difference(ABFileImporter.ABStandardProcessColumns.union(ABFileImporter.ABStandardBiosphereColumns))
        for scen in scen_cols:
            error = data.loc[(data[scen] == '#DIV/0!') | (data[scen] == '#VALUE!')]
            if not error.empty:
                msg = "Error with values for the exchanges between {} and {}".format(data.loc[0,'from activity name'], data.loc[0, 'to activity name'])
                raise ExchangeErrorValues(msg)

    @staticmethod
    def check_duplicates(data: pd.DataFrame, index: list=['to key', 'from key', 'flow type']) -> pd.DataFrame:
        """
        Checks three fields to identify whether a scenario difference file contains duplicate exchanges:
        'from key', 'to key' and 'flow type'
        Produces a warning
        """
        duplicates = data.duplicated(index, keep=False)
        if duplicates.any():
            warning = ABPopup()
            msg = """
            Duplicates have been found in the provided file. The Activity Browser cannot handle duplicate entries in the scenario files. Duplicate entries are discarded, only the last found instance of a duplicated entry will be used.
            
            If you want to proceed without changing the file contents please press 'ok', otherwise press 'cancel'.
            """
            warning.dataframe(data.loc[duplicates], index)
            response = warning.abWarning('Duplicate flow exchanges', msg, QMessageBox.Ok, QMessageBox.Cancel)
            if response == warning.Cancel:
                return None
            return data.drop_duplicates(index, keep='last', inplace=False)
        return data

    @staticmethod
    def check_activity_keys(data: pd.DataFrame) -> pd.DataFrame:
        """
        Checks the whether the activities have a key, for those activities without a key the AB metadata dataframes are
        used with the available information to identify the keys. Dataframes are loaded into the AB metadata if not
        already present. If activities are identified within the dataframe that do not occur within the metadata
        dataframes then a popup message is created that allows the user to save the interim dataframe.
        Arguments:
        data: pd.DataFrame = the dataframe constructed from the Scenario Difference File
        Returns:
        a scenario dataframe with all keys present, or throws a ABError
        """
        no_key_df = data.loc[data['from key'].isna() | data['to key'].isna()]

        # If all the keys are present then we can return the complete dataframe
        if no_key_df.shape[0] == 0:
            return data
        databases = set(no_key_df['from database'].to_list() + no_key_df['to database'].to_list())
        # need to append the flow direction (all except 'activity name' are compatible with brightway)
        metafields = ['name', 'categories', 'reference product', 'location']
        AB_metadata.add_metadata(databases)
        FIELDS = [pd.Index(['to activity name', 'to categories', 'to reference product', 'to location']),
                 pd.Index(['from activity name', 'from categories', 'from reference product', 'from location'])]
        keys = ['from key', 'to key']
        metadata = AB_metadata.dataframe
        critical = {'from database': [], 'from activity name': [], 'to database': [],
                    'to activity name': []}
        for idx in no_key_df.index:
            try:
                for i in [0,1]:
                    if isinstance(no_key_df.loc[idx,[FIELDS[i][1]]], float):
                        key = metadata[(metadata[metafields[0]] == no_key_df[FIELDS[i][0]]) &
                                       (metadata[metafields[2]] == no_key_df[FIELDS[i][2]]) &
                                       (metadata[metafields[3]] == no_key_df[FIELDS[i][3]])].copy()
                    else:
                        key = metadata[(metadata[metafields[0]] == no_key_df[FIELDS[i][0]]) &
                                       (metadata[metafields[1]] == no_key_df[FIELDS[i][1]])].copy()
                for j, col in enumerate([['from key', 'from database'], ['to key', 'to database']][i]):
                    no_key_df.loc[idx, no_key_df[col]] = (key['database'][0], key['code'][0]) if j == 0 else key['database'][0]
            except Exception:
                if len(critical['from database']) <= 5:
                    critical['from database'].append(no_key_df.loc[idx, 'from database'])
                    critical['from activity name'].append(no_key_df.loc[idx, 'from activity name'])
                    critical['to database'].append(no_key_df.loc[idx, 'to database'])
                    critical['to activity name'].append(no_key_df[idx, 'to activity name'])

        data.loc[no_key_df.index, keys] = no_key_df[keys]
        if critical['from database']:
            critical_message = ABPopup()
            critical_message.dataframe(pd.DataFrame(critical),
                                       ['from database', 'from activity name', 'to database', 'to activity name'])
            if len(critical['from database']) > 1:
                msg = f"Multiple activities in the exchange flows could not be linked. The first five of these are provided.\n\n" \
                      f"If you want to proceed with the import then press 'Ok' (doing so will enable you to save the dataframe\n" \
                      f"to either .csv, or .xlsx formats), otherwise press 'Cancel'"
                response = critical_message.abCritical("Activities not found", msg, QMessageBox.Save, QMessageBox.Cancel,
                                                       default=2)
            else:
                msg = f"An activity in the exchange flows could not been linked (See below for the activity).\n\nIf you want to" \
                      f"proceed with the import then press 'Ok' (doing so will enable you to save the dataframe to\n either .csv," \
                      f"or .xlsx formats), otherwise press 'Cancel'"
                response = critical_message.abCritical("Activity not found", msg, QMessageBox.Save, QMessageBox.Cancel,
                                                       default=2)
            if response == critical_message.Cancel:
                return pd.DataFrame({}, columns=data.columns)
            else:
                critical_message.save_dataframe(data)
                raise ABError(
                    "Incompatible Activities in the scenario file, unable to complete further checks on the file"
                )
        return data

    @staticmethod
    def fill_nas(data: pd.DataFrame) -> pd.DataFrame:
        """ Will replace NaNs in the dataframe with a string holding "NA" for the following subsection of columns:
            'from activity name', 'from reference product', 'to reference product', 'to location',
            'from location', 'to activity name', 'from database', 'to database', 'from unit', 'to unit',
            'from categories' and 'to categories'

            Note: How NaNs are treated depends on the 'flow type'
        """
        not_bio_cols = ABFileImporter.ABStandardProcessColumns.difference(ABFileImporter.ABScenarioColumnsErrorIfNA)
        bio_cols = ABFileImporter.ABStandardProcessColumns.union(ABFileImporter.ABStandardBiosphereColumns).difference(ABFileImporter.ABScenarioColumnsErrorIfNA)
        non_bio = data.loc[data.loc[:, 'flow type'] != 'biosphere'].fillna(dict.fromkeys(not_bio_cols, 'NA'))
        bio = data.loc[data.loc[:, 'flow type'] == 'biosphere'].fillna(dict.fromkeys(bio_cols, 'NA'))
        return pd.concat([non_bio, bio])

    @staticmethod
    def all_checks(data: pd.DataFrame, fields: set = None, scenario_names: list = None) -> None:
        if fields == None:
            fields = ABFeatherImporter.ABScenarioColumnsErrorIfNA
        if scenario_names == None:
            scenario_names = ABFeatherImporter.scenario_names(data)
        ABFileImporter.fill_nas(data)
        ABFileImporter.database_and_key_check(data)
        # Check all following uses of fields has the same requirements
        ABFileImporter.na_value_check(data, list(fields) + scenario_names)
        ABFileImporter.production_process_check(data, scenario_names)
        ABFileImporter.check_for_calculation_errors(data)

    @staticmethod
    def scenario_names(data: pd.DataFrame) -> list:
        return list(set(data.columns).difference(ABFileImporter.ABStandardProcessColumns.union(ABFileImporter.ABStandardBiosphereColumns)))


class ABFeatherImporter(ABFileImporter):
    def __init__(self):
        super(ABFeatherImporter, self).__init__(self)

    @staticmethod
    def read_file(path: Optional[Union[str, Path]], **kwargs):
        df = pd.read_feather(path)
        # ... execute code
        for i in ['to key', 'from key']:
            df.loc[:, i] = df.loc[:, i].apply(ast.literal_eval)
        return df


class ABCSVImporter(ABFileImporter):
    def __init__(self):
        super(ABCSVImporter, self).__init__(self)

    @staticmethod
    def read_file(path: Optional[Union[str, Path]], **kwargs):
        if 'separator' in kwargs:
            separator = kwargs['separator']
        else:
            separator = ";"
        df = pd.read_csv(path, compression='infer', sep=separator, index_col=False,
                         converters={'from key': ast.literal_eval, 'to key': ast.literal_eval})
        # ... execute code
        return df
