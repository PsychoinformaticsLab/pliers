from six import with_metaclass
from abc import ABCMeta, abstractmethod
from featurex.transformers import Transformer
import pandas as pd

__all__ = ['api', 'audio', 'google', 'image', 'text', 'video']


class Extractor(with_metaclass(ABCMeta, Transformer)):
    ''' Base class for Converters.'''

    def extract(self, stim, *args, **kwargs):
        return self._extract(stim, *args, **kwargs)

    @abstractmethod
    def _extract(self, stim):
        pass

    def _transform(self, stim, *args, **kwargs):
        return self.extract(stim, *args, **kwargs)


class ExtractorResult(object):

    def __init__(self, data, stim, transformer, features=None, onsets=None,
                 durations=None):
        self.data = data
        self.stim = stim
        self.transformer = transformer
        self.onsets = onsets
        self.features = features
        if durations is None:
            durations = np.nan
        if not isinstance(durations, (list, tuple)):
            durations = [durations] * len(self.data)
        self.durations = durations

    def to_df(self):
        df = pd.DataFrame(self.data)
        if self.features is not None:
            df.columns = self.features
        if self.onsets is not None:
            df.index = self.onsets
        df.insert(0, 'duration', self.durations)
        return df

    @classmethod
    def merge_stims(cls, results):
        pass

    def merge_features(cls, results):
        pass

def merge_results(results, extractor_names=True, stim_names=True,
                  by='infer'):
    dfs = [res.to_df() for res in results]

    # Stims merged along rows, features along columns

    # Verify that all elements can be merged--either number of rows must be
    # identical for all DFs, or onsets must be specified
