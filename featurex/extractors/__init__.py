from six import with_metaclass
from abc import ABCMeta, abstractmethod
from featurex.transformers import Transformer
import pandas as pd
import numpy as np
from collections import defaultdict

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

    def __init__(self, data, stim, extractor, features=None, onsets=None,
                 durations=None):
        self.data = data
        self.stim = stim
        self.extractor = extractor
        self.features = features
        self.onsets = onsets if onsets is not None else np.nan
        self.durations = durations if durations is not None else np.nan

    def to_df(self):
        df = pd.DataFrame(self.data)
        if self.features is not None:
            df.columns = self.features
        df.insert(0, 'duration', self.durations)
        df.insert(0, 'onset', self.onsets)
        return df

    @classmethod
    def merge_features(cls, results, extractor_names=True, stim_names=True):
        ''' Merge a list of ExtractorResults bound to the same Stim into a
        single DataFrame.

        Args:
            results (list, tuple): A list of ExtractorResult instances to merge
            extractor_names (bool): if True, stores the associated Extractor
                names in the top level of the column MultiIndex.
            stim_names (bool): if True, stores the associated Stim names in the
                top level of the row MultiIndex.
        '''

        # Make sure all ExtractorResults are associated with same Stim.
        stims = set([r.stim for r in results])
        dfs = [r.to_df() for r in results]
        if len(stims) > 1:
            raise ValueError("merge_features() can only be called on a set of "
                             "ExtractorResults associated with the same Stim.")

        keys = [r.extractor.name for r in results] if extractor_names else None

        # If onsets are all NaN
        if all([r['onset'].isnull().all() for r in dfs]):
            if len(set([len(r) for r in dfs])) > 1:
                raise ValueError("If ExtractorResults do not specify onsets, "
                                 "all ExtractorResults to merge must have "
                                 "identical numbers of rows.")
            dfs = [r.drop(['onset'], axis=1) for r in dfs]
            result = pd.concat(dfs, axis=1, keys=keys)
            result.insert(0, 'onset', np.nan)

        # If onsets are specified
        elif all([r['onset'].notnull().all() for r in dfs]):
            dfs = [r.set_index('onset').sort_index() for r in dfs]
            result = pd.concat(dfs, axis=1, keys=keys).reset_index()

        else:
            raise ValueError("To merge a list of ExtractorResults, all "
                             "instances must either contain onsets, or lack "
                             "onsets and have the same number of rows. It is "
                             "not possible to merge mismatched instances.")

        durations = result.xs('duration', level=1, axis=1)
        if durations.apply(lambda x: x.nunique()<=1, axis=1).all():
            result = result.drop('duration', axis=1, level=1)

        if stim_names:
            result['stim_name'] = list(stims)[0].name
            result.set_index('stim_name', append=True, inplace=True)

        return result

    @classmethod
    def merge_stims(cls, results, stim_names=True):
        pass


def merge_results(results, extractor_names=True, stim_names=True):
    ''' Merges a list of ExtractorResults instances and returns a pandas DF.
    Args:
        results (list, tuple): A list of ExtractorResult instances to merge.
        extractor_names (bool): if True, stores the associated Extractor
            names in the top level of the column MultiIndex.
        stim_names (bool): if True, stores the associated Stim names in the
            top level of the row MultiIndex.

    Returns: a pandas DataFrame with features concatenated along the column
        axis and stims concatenated along the row axis.
    '''

    stims = defaultdict(list)

    for r in results:
        stims[r.stim.name].append(r)

    # First concatenate all features separately for each Stim
    for k, v in stim_lists.items():
        stims[k] = ExtractorResult.merge_features(v, extractor_names)

    # Now concatenate all Stims
    stims = list(stims.values())
    return stims[0] if len(stims) == 1 else ExtractorResult.merge_stims(stims)
