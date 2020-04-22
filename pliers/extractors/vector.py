'''
Extractors that operate primarily or exclusively on Vector stimuli.
'''
from pliers.stimuli.vector import VectorStim
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.utils import listify
import scipy
import numpy as np
import pandas as pd
from collections.abc import Iterable
import logging

class VectorMetricExtractor(Extractor):
    ''' Extracts summary metrics from 1D-array using numpy, scipy or custom 
        functions
    Args:
        functions (str, functions or list): function or list of numpy, scipy, 
        or custom functions to be applied to 1-dimensional numpy arrays.
            Functions can be passed directly (e.g. passing scipy.stats.entropy) if 
            the package/module they belong to has been imported, or as strings 
            (e.g. 'np.mean'). Custom functions returning integers or iterables 
            can also be passed, either directly, or as strings (evaluated via 
            eval() method).
        var_names (optional): list of custom alias names for each metric
        kwargs: named arguments for function call
    ''' 

    _input_type = VectorStim
    _log_attributes = ('functions',)

    def __init__(self, functions=['np.mean', 'np.std', 'np.max', 'np.min'], 
                 var_names=None, **kwargs):
        functions = listify(functions)
        if var_names is not None:
            if len(var_names) != len(functions):
                raise ValueError('Length or var_names must match number of '
                                 'functions')
        for idx, f in enumerate(functions):
            if isinstance(f, str):
                f = f.replace('numpy', 'np')
                try:
                    f_list = f.split('.')
                    if len(f_list) > 1 and 'lambda' not in f:
                        functions[idx] = getattr(eval('.'.join(f_list[:-1])), 
                                                f_list[-1])
                    else:
                        functions[idx] = eval(f)
                except:
                    raise ValueError(f'{f} is not a valid function')
        if var_names is None:
            var_names = [f.__name__ for f in functions]
        self.var_names = var_names

        self.functions = functions
        self.kwargs = kwargs
        super(VectorMetricExtractor, self).__init__()
        
    def _extract(self, stim):
        outputs = []
        for f in self.functions:
            metrics = f(stim.array, **self.kwargs)
            if isinstance(metrics, Iterable):
                metrics = np.array(metrics)
            outputs.append(metrics)
        return ExtractorResult([outputs], stim, self, self.var_names)
            
            
