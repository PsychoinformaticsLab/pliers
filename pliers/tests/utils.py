from os.path import dirname, join
from pliers.stimuli import ImageStim
from pliers.extractors.base import Extractor, ExtractorResult
import numpy as np
from copy import deepcopy


def get_test_data_path():
    """Returns the path to test datasets """
    return join(dirname(__file__), 'data')


class DummyExtractor(Extractor):

    ''' A dummy Extractor class that always returns random values when
    extract() is called. Can set the extractor name inside _extract() to
    facilitate testing of results merging etc. '''
    _input_type = ImageStim
    _log_attributes = ('param_A', 'param_B')

    def __init__(self, param_A=None, param_B='pie'):
        super(DummyExtractor, self).__init__()
        self.param_A = param_A
        self.param_B = param_B

    def _extract(self, stim, name=None, n_rows=100, n_cols=3, max_time=1000):
        data = np.random.randint(0, 1000, (n_rows, n_cols))
        onsets = np.random.choice(n_rows*2, n_rows, False)
        if name is not None:
            self.name = name
        return ExtractorResult(data, stim, deepcopy(self), onsets=onsets)
