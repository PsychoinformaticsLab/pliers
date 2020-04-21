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
        functions (str, functions or list): function or list of functions
            that can be applied to 1-dimensional numpy arrays.
            Numpy functions must be specified as np.<function_name>, or
                as string (e.g. 'mean')
            Scipy functions must be specified as scipy.<module/function_name>,
                e.g. scipy.stats.entropy.
        var_names (optional) = custom list of names for the outcome variables
        kwargs: named arguments for function call
    ''' 

    _input_type = VectorStim
    _log_attributes = ('functions',)

    def __init__(self, functions, var_names=None, **kwargs):
        functions = listify(functions)
        for idx, f in enumerate(functions):
            if type(f) == str:
                f = f.replace('numpy', 'np')
                try:
                    f_list = f.split('.')
                    if len(f_list) > 1:
                        functions[idx] = getattr(eval('.'.join(f_list[:-1])), 
                                                f_list[-1])
                    else:
                        functions[idx] = eval(f)
                except: 
                    raise ValueError(f'{f} is not a valid numpy or '
                                     'scipy function')
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
            
            