'''
Extractors that operate primarily or exclusively on Vector stimuli.
'''
from pliers.stimuli.vector import VectorStim
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.utils import listify
import scipy
import numpy as np
import pandas as pd

class VectorMetricExtractor(Extractor):
    ''' Extracts summary metrics from 1D-array using numpy, scipy or custom 
        functions
    Args:
        functions (function, str or list): function or list of functions
            that can be applied to 1-dimensional numpy arrays.
            Numpy functions must be specified as np.<function_name>, or
                as string (e.g. 'mean')
            Scipy functions must be specified as scipy.<module/function_name>,
                e.g. scipy.stats.entropy.
        var_names (optional) = custom list of names for the outcome variables
        kwargs: named arguments for function call
    ''' 

    _input_type = VectorStim
    _log_attributes = ('functions')

    def __init__(self, functions, var_names=None, **kwargs):
        functions = listify(functions)
        for idx, f in enumerate(functions):
            if type(f) == str:
                try:
                    functions[idx] = eval(f)
                except:
                    raise ValueError(f'{f} is not a valid function')
        if var_names is not None:
            self.var_names = var_names
        else:
            self.var_names = [f.__name__ for f in functions]
        self.functions = functions
        self.kwargs = kwargs
        
    def extract(self, stim):
        outputs = []
        features = []
        for idx, f in enumerate(self.functions):
            metrics = f(stim.array, self.kwargs)
            metrics = listify(metrics)
            outputs += metrics
            if len(self.var_names[idx]) == len(metrics):
                features += self.var_names[idx]
            elif len(self.var_names[i] == 1):
                features += [self.var_names[idx] + str(i) for i in range(len(metrics))]
            else:
                raise IndexError(f'Function {f.__name__} outputs {len(metrics)},'
                                 f'but {len(self.var_names[idx])} scores were '
                                 f'provided: ({self.var_names[idx]})')
        return ExtractorResult(outputs, stim, self, features, stim.onset,
                               stim.duration, stim.order)
            
            