'''
Extractors that operate on Miscellaneous Stims.
'''

from pliers.stimuli.misc import SeriesStim
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.utils import listify
import scipy
import numpy as np
import pandas as pd
from collections.abc import Iterable
from importlib import import_module
import logging

class MetricExtractor(Extractor):
    ''' Extracts summary metrics from 1D-array using numpy, scipy or custom 
        functions
    Args:
        functions (str, functions or list): function or string referring to absolute
           import path for a function (e.g. 'numpy.mean'). Function must operate 
           on 1-dimensional numpy arrays and return a scalar. A list of 
           functions or import strings may also be passed.
        var_names (list): optional list of custom alias names for each metric
        subset_idx (list): subset of Series indices to compute metric on.
        kwargs: named arguments for function call
    ''' 

    _input_type = SeriesStim
    _log_attributes = ('functions', 'subset_idx')

    def __init__(self, functions=None, var_names=None, 
                 subset_idx=None, **kwargs):
        functions = listify(functions)
        if var_names is not None:
            var_names = listify(var_names)
            if len(var_names) != len(functions):
                raise ValueError('Length or var_names must match number of '
                                 'functions')
        for idx, f in enumerate(functions):
            if isinstance(f, str):
                try:
                    f_mod, f_func = f.rsplit('.', 1)
                    functions[idx] = getattr(import_module(f_mod),
                                             f_func)
                except:
                    raise ValueError(f"{f} is not a valid function")
        if var_names is None:
            var_names = [f.__name__ for f in functions]
        self.var_names = var_names

        self.functions = functions
        self.kwargs = kwargs
        self.subset_idx = subset_idx
        super(MetricExtractor, self).__init__()
        
    def _extract(self, stim):
        outputs = []
        if self.subset_idx is not None:
            idx_diff = set(self.subset_idx) - set(stim.data.index)
            idx_int = set(self.subset_idx) & set(stim.data.index)
            if idx_diff:
                logging.warning(f'{idx_diff} not in index.')
            if not idx_int:
                raise ValueError('No valid index')
            series = stim.data[idx_int]
        else:
            series = stim.data
        for f in self.functions:
            metrics = f(series, **self.kwargs)
            if isinstance(metrics, Iterable):
                metrics = np.array(metrics)
            outputs.append(metrics)
        return ExtractorResult([outputs], stim, self, self.var_names)
            
            
