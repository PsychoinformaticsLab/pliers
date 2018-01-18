''' Base Extractor class and associated functionality. '''

from collections import defaultdict
from abc import ABCMeta, abstractmethod
from six import with_metaclass
import pandas as pd
import numpy as np
from pliers.transformers import Transformer
from pliers.utils import isgenerator, flatten


class Extractor(with_metaclass(ABCMeta, Transformer)):

    ''' Base class for all pliers Extractors.'''

    def __init__(self):
        super(Extractor, self).__init__()

    def transform(self, stim, *args, **kwargs):
        result = super(Extractor, self).transform(stim, *args, **kwargs)
        return list(result) if isgenerator(result) else result

    @abstractmethod
    def _extract(self, stim):
        pass

    def _transform(self, stim, *args, **kwargs):
        return self._extract(stim, *args, **kwargs)

    def plot(self, result, stim=None):
        raise NotImplementedError("No plotting method is defined for class "
                                  "%s." % self.__class__.__name__)


class ExtractorResult(object):

    ''' Stores feature data produced by an Extractor.

    Args:
        data (ndarray, iterable): Extracted feature values. Either an ndarray
            or an iterable. Can be either 1-d or 2-d.
        stim (Stim): The input Stim object from which features were extracted.
        extractor (Extractor): The Extractor object used in extraction.
        features (list, ndarray): Optional names of extracted features. If
            passed, must have as many elements as there are columns in data.
        onsets (list, ndarray): Optional iterable giving the onsets of the
            rows in data. Length must match the input data.
        durations (list, ndarray): Optional iterable giving the durations
            associated with the rows in data.
        raw: The raw result (net of any containers or overhead) returned by
            the underlying feature extraction tool. Can be an object of any
            type.
    '''

    def __init__(self, data, stim, extractor, features=None, onsets=None,
                 durations=None, raw=None):

        if data is None and raw is None:
            raise ValueError("At least one of 'data' and 'raw' must be a "
                             "value other than None.")

        self.stim = stim
        self.extractor = extractor
        self.features = features
        self.raw = raw
        self._history = None

        # Eventually, the goal is to make raw mandatory, and always
        # generate the .data property via calls to to_array() or to_df()
        # implemented in the Extractor. But to avoid breaking the API without
        # warning, we provide a backward-compatible version for the time being.
        if self.raw is not None and hasattr(self.extractor, 'to_array'):
            self.data = self.extractor.to_array(self)
        else:
            self.data = np.array(data)

        if onsets is None:
            onsets = stim.onset
        self.onsets = onsets if onsets is not None else np.nan

        if durations is None:
            durations = stim.duration
        self.durations = durations if durations is not None else np.nan

    def to_df(self, timing=True, metadata=False):
        ''' Convert current instance to a pandas DatasFrame.

        Args:
            timing (bool): If True, adds columns for event onset and duration.
                Note that these columns will be added even if there are no
                valid values in the current object (NaNs will be inserted).
            metadata (bool): If True, adds columns for key metadata (including
                the name, filename, class, history, and source file of the
                Stim).
        Returns:
            A pandas DataFrame.
        '''

        if hasattr(self.extractor, '_to_df'):
            df = self.extractor._to_df(self)
        else:
            features = self.features
            if features is None:
                features = ['feature_%d' % (i + 1)
                            for i in range(self.data.shape[1])]
            df = pd.DataFrame(self.data, columns=features)

        # For results with more than one object (e.g., multiple faces in a
        # single image), add an object_id column.
        ???

        if timing:
            n = len(df)
            df.insert(0, 'duration', np.repeat(self.durations, n))
            df.insert(0, 'onset', np.repeat(self.onsets, n))
        if metadata:
            df['stim_name'] = self.stim.name
            df['class'] = self.stim.__class__.__name__
            df['filename'] = self.stim.filename
            df['history'] = str(self.stim.history)
            df['source_file'] = self.history.to_df().iloc[0].source_file
        return df

    @property
    def history(self):
        ''' Returns the transformation history for the input Stim. '''
        return self._history

    @history.setter
    def history(self, history):
        self._history = history

    @classmethod
    def merge_features(cls, results, metadata=True, extractor_names=True,
                       flatten_columns=False):
        ''' Merge a list of ExtractorResults bound to the same Stim into a
        single DataFrame.

        Args:
            results (list, tuple): A list of ExtractorResult instances to merge
            extractor_names (bool): if True, stores the associated Extractor
                names in the top level of the column MultiIndex.
            metadata (bool): if True, stores all ExtractorResult metadata
            flatten_columns (bool): if True, flattens the resultant column
                MultiIndex such that feature columns are in the format
                <extractor class>_<feature name>
        '''

        # Make sure all ExtractorResults are associated with same Stim.
        stims = set([r.stim.name for r in results])
        dfs = [r.to_df(metadata=metadata) for r in results]
        if len(stims) > 1:
            raise ValueError("merge_features() can only be called on a set of "
                             "ExtractorResults associated with the same Stim.")

        keys = [r.extractor.name for r in results] if extractor_names else None
        extra_columns = ['onset', 'duration', 'class', 'filename', 'history',
                         'stim_name', 'source_file']

        # If onsets are all NaN
        if all([r['onset'].isnull().all() for r in dfs]):
            if len(set([len(r) for r in dfs])) > 1:
                raise ValueError("If ExtractorResults do not specify onsets, "
                                 "all ExtractorResults to merge must have "
                                 "identical numbers of rows.")
            feature_dfs = [r.drop(extra_columns, axis=1) for r in dfs]
            result = pd.concat(feature_dfs, axis=1, keys=keys)
            result.insert(0, 'onset', np.nan)

        # If onsets are specified
        elif all([r['onset'].notnull().all() for r in dfs]):
            feature_dfs = [r.set_index('onset').sort_index() for r in dfs]
            feature_dfs = [r.drop(extra_columns, axis=1, errors='ignore') for r in feature_dfs]
            result = pd.concat(feature_dfs, axis=1, keys=keys).reset_index()

        else:
            raise ValueError("To merge a list of ExtractorResults, all "
                             "instances must either contain onsets, or lack "
                             "onsets and have the same number of rows. It is "
                             "not possible to merge mismatched instances.")

        extra_columns.remove('onset')
        if not metadata:
            extra_columns = ['duration']
        for col in extra_columns:
            result.insert(0, col, dfs[0][col][0])

        result = result.sort_values(['onset']).reset_index(drop=True)
        if flatten_columns:
            result.columns = ['_'.join(str(lvl) for lvl in col).strip('_') for col in result.columns.values]
        return result

    @classmethod
    def merge_stims(cls, results):
        results = [r.to_df(metadata=True) if isinstance(
            r, ExtractorResult) else r for r in results]
        return pd.concat(results, axis=0).sort_values('onset').reset_index(drop=True)


# def merge_results(results, format='long', **merge_feature_args):
#     ''' Merges a list of ExtractorResults instances and returns a pandas DF.

#     Args:
#         results (list, tuple): A list of ExtractorResult instances to merge.
#         format (str): Format to return the data in. Can be either 'wide' or
#             'long'. In the wide case, every extracted feature is a column,
#             and every Stim is a row. In the long case, every row contains a
#             single Stim/Extractor/feature combination.
#         merge_feature_args (kwargs): Additional argument settings to use
#             when merging across features.

#     Returns: a pandas DataFrame with features concatenated along the column
#         axis and stims concatenated along the row axis.
#     '''

#     # Flatten list recursively
#     results = flatten(results)

#     stims = defaultdict(list)

#     for r in results:
#         stims[hash(r.stim)].append(r)

#     # First concatenate all features separately for each Stim
#     for k, v in stims.items():
#         stims[k] = ExtractorResult.merge_features(v, **merge_feature_args)

#     # Now concatenate all Stims
#     stims = list(stims.values())
#     return stims[0] if len(stims) == 1 else \
#         ExtractorResult.merge_stims(stims)


def merge_results(results, format='long', timing='auto', metadata=True,
                  extractor_names=True, flatten_columns=False):
    ''' Merges a list of ExtractorResults instances and returns a pandas DF.

    Args:
        results (list, tuple): A list of ExtractorResult instances to merge.
        format (str): Format to return the data in. Can be either 'wide' or
            'long'. In the wide case, every extracted feature is a column,
            and every Stim is a row. In the long case, every row contains a
            single Stim/Extractor/feature combination.
        timing (bool, str): Whether or not to include columns for onset and
            duration.
        extractor_names (bool): if True, stores the associated Extractor
            names as a variable.
        metadata (bool): if True, stores all ExtractorResult metadata
        flatten_columns (bool): if True, flattens the resultant column
            MultiIndex such that feature columns are in the format
            <extractor class>_<feature name>

    Returns: a pandas DataFrame. For format details, see 'format' argument.
    '''

    _timing = True if timing == 'auto' else timing
    dfs = [r.to_df(timing=_timing, metadata=metadata) for r in results]

    extra_columns = ['class', 'filename', 'history', 'stim_name',
                     'source_file']

    data = pd.concat(dfs, axis=0)

    if timing == 'auto':
        if data['onset'].isnull().all():
            data = data.drop(['onset', 'duration'], axis=1)

    extra_columns.remove('onset')
    for col in extra_columns:
        result.insert(0, col, dfs[0][col][0])

    result = result.sort_values(['onset']).reset_index(drop=True)
    if flatten_columns:
        result.columns = ['_'.join(str(lvl) for lvl in col).strip('_') for col in result.columns.values]
    return result

