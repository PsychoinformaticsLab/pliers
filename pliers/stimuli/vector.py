import numpy as np
import pandas as pd
import requests
import json
from pliers.utils import attempt_to_import, verify_dependencies
from pliers.stimuli.base import Stim


class VectorStim(Stim):
    ''' Vector stimulus (1-D numpy array)

    Args:
        array (np.ndarray, list or pd.Series): Vector of values. Can be list, 
            array or pandas Series. Gets converted to 1-D numpy array.
        labels (list): List of labels to which probability values refer, if 
            that applies.
        filename (str): Path to tsv file, if array should be read from file. 
            Must be tsv file, with a label column (label_column) and a value 
            column (data_column). 
        url (str): url to read from, if filename is None. Must point to tsv 
            file with a value column (data_column) and a label column 
            (label_column, optional).
        data_column (str): If filename or url is defined, defines column to 
            read in as array.
        label_column (str): Optional. If filename or url is defined, defines 
            column where labels can be found.
        onset (float): Optional onset of the event the probability 
            distribution refers to.
        duration (float): Optional duration of the event the vector refers to.
        order (int): Optional sequential index of the event the vector refers 
            to within some broader context.
        name (str): Optional name to give to the Stim instance. If None is
            provided, the name will be derived from the filename if one is
            defined. If no filename is defined, name will be an empty string.
        sort_data (str): 'descending' or 'ascending'. Sorts array (and labels)
            in ascending or descending order.
    '''

    _default_file_extension='.txt'

    def __init__(self, array=None, labels=None, filename=None,
        data_column='value', label_column='label', onset=None, duration=None,
        order=None, name=None, sort_data=None, url=None):

        tsv = filename or url or None
        if tsv is not None:
            df = pd.read_csv(tsv, sep='\t')
            array = np.array(df[data_column].values)
            if label_column is not None:
                labels = list(df[label_column])

        array = np.array(array).squeeze()
        if len(array.shape) != 1:
            raise ValueError('Array must be one-dimensional')

        if sort_data in ['ascending', 'descending']:
            array, labels = self._sort(array, labels, sort_data)

        if labels is not None:
            self.labels = labels
        else: 
            self.labels = [str(idx) for idx in range(array.shape[0])]
        self.array = array

        if len(self.labels) != self.array.shape[0]:
            raise ValueError('Label and data must be of the same length')
        super(VectorStim, self).__init__(filename, onset, duration, order, name)

    def _sort(self, array, labels, sort_data):
        idxs = np.argsort(array)
        array = array[idxs]
        labels = [labels[i] for i in idxs]  
        if sort_data == 'descending':
            labels = labels[::-1]
            array = array[::-1]
        return array, labels

    @property
    def data(self):
        return self.array

    def save(self, path):
        df = pd.DataFrame(data=zip(self.labels, self.data),
                          columns=['label', 'value'])
        df.to_csv(path, sep='\t')
