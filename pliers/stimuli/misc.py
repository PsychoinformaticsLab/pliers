"""Miscellaneous Stim classes."""

import numpy as np
import pandas as pd

from .base import Stim


class SeriesStim(Stim):
    '''Represents a pandas Series as a pliers Stim.

    Args:
        data (dict, pd.Series, array-like): A dictionary, pandas Series, or any
            other iterable (e.g., list or 1-D numpy array) that can be coerced
            to a pandas Series.
        filename (str, optional): Path or URL to data file. Must be readable
            using pd.read_csv().
        onset (float): Optional onset of the SeriesStim (in seconds) with
            respect to some more general context or timeline the user wishes
            to keep track of.
        duration (float): Optional duration of the SeriesStim, in seconds.
        order (int): Optional order of stim within some broader context.
        url (str): Optional URL to read data from. Must be readable using
            pd.read_csv().
        column (str): If filename or url is passed, defines the name of the
            column in the data source to read in as data.
        name (str): Optional name to give the SeriesStim instance. If None
            is provided, the name will be derived from the filename if one is
            defined. If no filename is defined, name will be an empty string.
        pd_args: Optional keyword arguments passed onto pd.read_csv() (e.g., 
            to control separator, header, etc.).
    '''

    def __init__(self, data=None, filename=None, onset=None, duration=None,
                 order=None, url=None, column=None, name=None, **pd_args):

        if data is None:
            if filename is None and url is None:
                raise ValueError("No data provided! One of the data, filename,"
                                 "or url arguments must be passed.")
            source = filename or url
            data = pd.read_csv(source, squeeze=True, **pd_args)
            if isinstance(data, pd.DataFrame):
                if column is None:
                    raise ValueError("Data source contains more than one "
                                    "column; please specify which column to "
                                    "use by passing the 'column' argument.")
                data = data.loc[:, column]
        
        data = pd.Series(data)
        self.data = data
        super().__init__(filename, onset, duration, order, name)

    def save(self, path):
        self.data.to_csv(path)
