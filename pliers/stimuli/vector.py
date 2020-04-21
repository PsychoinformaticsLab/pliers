import numpy as np
import pandas as pd
import requests
import json
from pliers.utils import attempt_to_import, verify_dependencies
from .base import Stim


class VectorStim(Stim):
    ''' Vector stimulus (1-D numpy array)

    Args:
        filename (str): Path to input file, if one exists. Can be a tsv file 
            with probability distribution values in one column, or a json file
            with labels as keys and probability values as values.
        data_column (str): If filename is defined, defines column to read
            in as probability distribution
        label_column (str): If filename is defined, defines columns where 
            labels can be found.
        array (np.ndarray, list or pd.Series): Vector of values. Can be list, 
            array or pandas Series. Gets converted to 1-D numpy array.
        labels (list): List of labels to which probability values refer, if 
            that applies.
        data_dict (dict): If defined, overwrites array and labels. Keys are 
            passed to labels argument, values are passed to array argument.
        onset (float): Optional onset of the event the probability 
            distribution refers to.
        duration (float): Optional duration of the event the probability 
            distribution refers to.
        order (int): Optional sequential index of the event the probability 
            distribution refers to within some broader context.
        name (str): Optional name to give to the Stim instance. If None is
            provided, the name will be derived from the filename if one is
            defined. If no filename is defined, name will be an empty string.
        sort_data (str): 'descending' or 'ascending'. Sorts array (and labels)
            in ascending or descending order.
        url (str): Optional url to read contents from. Must be json readable
            dictionary with labels as keys and values as probability values.
    '''

    def __init__(self, filename=None, data_column=None, label_column=None, 
        array=None, labels=None, data_dict=None, onset=None, duration=None, 
        order=None, name=None, sort_data=None, url=None):
    
        if filename is not None:
            if filename.endswith('.json'):
                data_dict = json.loads(filename)
            else:
                array = pd.read_csv(filename, sep='\t')
                if data_column is not None:
                    array = np.array(array[data_column].values)
                if label_column is not None:
                    labels = list(array[label_column])
        elif url is not None:
            data_dict = requests.get(url).json()
        
        if data_dict:
            labels = list(data_dict.keys())
            array = np.array(list(data_dict.values()))

        array = np.array(array).squeeze()
        if len(array.shape) != 1:
            raise ValueError('')

        if sort_data in ['ascending', 'descending']:
            array, labels = self._sort(array, labels, sort_data)

        if labels is not None:
            self.labels = labels
        else: 
            self.labels = [str(idx) for idx in range(array.shape[0])]
        self.array = array
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

    def save(self, path, format='json'):
        if format == 'json':
            with open(path, 'w') as f:
                json.dump(zip(self.labels, self.data), f)
        else:
            df = pd.DataFrame(data=zip(self.labels, self.data),
                              columns=['label', 'value'])
            df.to_csv(path, sep='\t')
