# -*- coding: utf-8 -*-


class ABError(Exception):
    """To be used as a generic Activity-Browser Error that will not lead to the AB crashing out"""


class ABWarning(Warning):
    """To be used as a generic Activity-Browser Warning"""


class ImportCanceledError(ABError):
    """Import of data was cancelled by the user."""
    pass


class LinkingFailed(ABError):
    """Unlinked exchanges remain after relinking."""
    pass


class IncompatibleDatabaseNamingError(ABError):
    """Database and keys do not match."""
    pass


class ActivityProductionValueError(ABError):
    """Production value for an activity == 0"""
    pass


class InvalidSDFEntryValue(ABError):
    """NA values found for data type that cannot hold \"NA\"."""
    pass


class ExchangeErrorValues(ABError):
    """In Brightway2 if there is an error in an exchange calculation the 'amount' field is not available for the
        Exchange"""
    pass


class ReferenceFlowValueError(ABWarning):
    """While a user can technically perform a calculation with the reference flows all set to 0, such a calculation
     makes no logical sense and will lead to downstream errors (due to 0 results)."""
    pass


class DuplicatedScenarioExchangeWarning(ABWarning):
    """Will warn the user that a loaded scenario table contains duplicate exchanges. Only the last added exchange value
    will be used."""
    pass