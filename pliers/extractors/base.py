''' Base Extractor class and associated functionality. '''

from abc import ABCMeta, abstractmethod
from six import with_metaclass
import pandas as pd
import numpy as np
from pliers.transformers import Transformer
from pliers.utils import isgenerator, flatten
from pandas.api.types import is_numeric_dtype


class Extractor(with_metaclass(ABCMeta, Transformer)):

    ''' Base class for all pliers Extractors.'''

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
        data (ndarray, iterable): Extracted feature data. Either an ndarray
            (1-d or 2-d), an iterable, or a raw result. If a raw result is
            passed, the source Extractor must implement _to_df().
        stim (Stim): The input Stim object from which features were extracted.
        extractor (Extractor): The Extractor object used in extraction.
        features (list, ndarray): Optional names of extracted features. If
            passed, must have as many elements as there are columns in data.
        onsets (list, ndarray): Optional iterable giving the onsets of the
            rows in data. Length must match the input data.
        durations (list, ndarray): Optional iterable giving the durations
            associated with the rows in data.
        orders (list, ndarray): Optional iterable giving the integer orders
            associated with the rows in data.
        raw: The raw result (net of any containers or overhead) returned by
            the underlying feature extraction tool. Can be an object of any
            type.
    '''

    def __init__(self, data, stim, extractor, features=None, onsets=None,
                 durations=None, orders=None):
        self._data = data
        self.stim = stim
        self.extractor = extractor
        self.features = features
        self._history = None
        self.onset = onsets
        self.duration = durations
        self.order = orders

    @property
    def raw(self):
        ''' Stores raw result of extraction, prior to postprocessing done
        in to_df(). '''
        return self._data if hasattr(self.extractor, '_to_df') else None

    @property
    def data(self):
        ''' Creates a DataFrame with default arguments '''
        return self.to_df()

    def to_df(self, timing=True, metadata=False, format='wide',
              extractor_name=False, object_id=True, **to_df_kwargs):
        ''' Convert current instance to a pandas DatasFrame.

        Args:
            timing (bool): If True, adds columns for event onset and duration.
                Note that these columns will be added even if there are no
                valid values in the current object (NaNs will be inserted).
                If 'auto', timing columns are only inserted if there's at least
                one valid (i.e., non-NaN) onset/order/duration.
            metadata (bool): If True, adds columns for key metadata (including
                the name, filename, class, history, and source file of the
                Stim).
            format (str): Format to return the data in. Can be either 'wide' or
                'long'. In the wide case, every extracted feature is a column,
                and every result object is in a row. In the long case, every
                row contains a single record/feature combination.
            extractor_name (bool): If True, includes the Extractor name as
                a column (in 'long' format) or index level (in 'wide' format).
            object_id (bool): If True, attempts to intelligently add an
                'object_id' column that differentiates between multiple objects
                in the results that may share onsets and durations (and would
                otherwise be impossible to distinguish). This frequently occurs
                for ImageExtractors that identify multiple target objects
                (e.g., faces) within a single ImageStim. In addition to boolean
                values, the special value 'auto' can be passed, in which case
                the object_id column will only be inserted if the resulting
                constant would be non-constant.

        Returns:
            A pandas DataFrame.
        '''

        # Ideally, Extractors should implement their own _to_df() class method
        # that produces a DataFrame in standardized format. Failing that, we
        # assume self._data is already array-like and can be wrapped in a DF.
        if hasattr(self.extractor, '_to_df'):
            df = self.extractor._to_df(self, **to_df_kwargs)
        else:
            features = self.features
            data = np.array(self._data)
            if features is None:
                features = ['feature_%d' % (i + 1)
                            for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=features)

        if hasattr(self, '_onsets'):
            onsets = np.array(self._onsets)
            onsets += 0.0 if self.onset is None else self.onset
        else:
            onsets = np.nan if self.onset is None else self.onset
        durations = getattr(self, '_durations', self.duration)
        durations = np.nan if durations is None else durations
        orders = getattr(self, '_orders', self.order)
        orders = np.nan if orders is None else orders

        # If any features clash with protected keys, append underscore
        protected = ['onset', 'order', 'duration', 'extractor', 'stim_name', \
                     'class', 'filename', 'history', 'source_file']
        df = df.rename(columns={k: k + '_' for k in protected})

        index_cols = []

        # Generally we leave it to Extractors to properly track the number of
        # objects returned in the result DF, using the 'object_id' column.
        # But in cases where the Extractor punt on this and_object_id=True, we
        # take our best guess. The logic is that we increment the object
        # counter for any row in the DF that cannot be uniquely distinguished
        # from other rows by onset and duration.
        if object_id and 'object_id' not in df.columns:
            index = pd.Series(onsets).astype(str) + '_' + \
                pd.Series(durations).astype(str)
            if object_id is True or (object_id == 'auto' and
                                     len(set(index)) < len(df)):
                ids = np.arange(len(df)) if len(index) == 1 \
                    else df.groupby(index).cumcount()
                df.insert(0, 'object_id', ids)
                index_cols = ['object_id']

        if timing is True or (timing == 'auto' and
                              (np.isfinite(durations).any() or
                               np.isfinite(orders).any())):
            df.insert(0, 'onset', onsets)
            df.insert(0, 'duration', durations)
            df.insert(0, 'order', orders)
            df = df.sort_values('onset').reset_index(drop=True)
            index_cols.extend(['onset', 'order', 'duration'])

        if format == 'long':
            df = df.melt(index_cols, var_name='feature')
            df = df.dropna(subset=['value'])

        if extractor_name:
            name = self.extractor.name
            if format == 'long':
                df['extractor'] = name
            else:
                df.columns = pd.MultiIndex.from_product([[name], df.columns])

        if metadata:
            df['stim_name'] = self.stim.name
            df['class'] = self.stim.__class__.__name__
            df['filename'] = self.stim.filename
            hist = '' if self.stim.history is None else str(self.stim.history)
            df['history'] = hist
            df['source_file'] = self.history.to_df().iloc[0].source_file
        return df

    @property
    def history(self):
        ''' Returns the transformation history for the input Stim. '''
        return self._history

    @history.setter
    def history(self, history):
        self._history = history


def merge_results(results, format='wide', timing=True, metadata=True,
                  extractor_names=True, object_id=True, aggfunc=None,
                  invalid_results='ignore', **to_df_kwargs):
    ''' Merges a list of ExtractorResults instances and returns a pandas DF.

    Args:
        results (list, tuple): A list of ExtractorResult instances to merge.
        format (str): Format to return the data in. Can be either 'wide' or
            'long'. In the wide case, every extracted feature is a column,
            and every Stim is a row. In the long case, every row contains a
            single Stim/Extractor/feature combination.
        timing (bool, str): Whether or not to include columns for onset,
            order, and duration.
        metadata (bool): if True, includes Stim metadata columns in the
            returned DataFrame. These columns include 'stim_name', 'class',
            'filename', 'history', and 'source_file'. Note that these values
            are often long strings, so the returned DF will be considerably
            larger.
        extractor_names (str, bool): How to handle extractor names when
            returning results. The specific behavior depends on whether format
            is 'long' or 'wide'. Valid values include:

                - 'prepend' or True: In both 'long' and 'wide' formats,
                  feature names will be prepended with the Extractor name
                  (e.g., "FaceExtractor#face_likelihood").
                - 'drop' or False: In both 'long' and 'wide' formats, extractor
                  names will be omitted entirely from the result. Note that
                  this can create feature name conflicts when merging results
                  from multiple Extractors, so is generally discouraged.
                - 'column': In 'long' format, extractor name will be included
                  as a separate column. Not valid for 'wide' format (and will
                  raise an error).
                - 'multi': In 'wide' format, a MultiIndex will be used for the
                  columns, with the first level of the index containing the
                  Extractor name and the second level containing the feature
                  name. This value is invalid if format='long' (and will raise
                  and error).

        object_id (bool): If True, attempts to intelligently add an
            'object_id' column that differentiates between multiple objects in
            the results that may share onsets/orders/durations (and would
            otherwise be impossible to distinguish). This frequently occurs for
            ImageExtractors that identify multiple target objects (e.g., faces)
            within a single ImageStim. Default is 'auto', which includes the
            'object_id' column if and only if it has a non-constant value.
        aggfunc (str, Callable): If format='wide' and extractor_names='drop',
            it's possible for name clashes between features to occur. In such
            cases, the aggfunc argument is passed onto pandas' pivot_table
            function, and specifies how to aggregate multiple values for the
            same index. Can be a callable or any string value recognized by
            pandas. By default (None), 'mean' will be used for numeric columns
            and 'first' will be used for object/categorical columns.
        invalid_results (str): Specifies desired action for treating elements
            of the passed in results argument that are not ExtractorResult
            objects. Valid values include:
                - 'ignore' will ignore them and merge the valid
                    ExtractorResults.
                - 'fail' will raise an exception on any invalid input


    Returns: a pandas DataFrame. For format details, see 'format' argument.
    '''

    results = flatten(results)

    _timing = True if timing == 'auto' else timing
    _object_id = True if object_id == 'auto' else object_id

    if extractor_names is True:
        extractor_names = 'prepend'
    elif extractor_names is False:
        extractor_names = 'drop'

    dfs = []
    for r in results:
        if isinstance(r, ExtractorResult):
            dfs.append(r.to_df(timing=_timing, metadata=metadata,
                               format='long', extractor_name=True,
                               object_id=_object_id, **to_df_kwargs))
        elif invalid_results == 'fail':
            raise ValueError("At least one of the provided results was not an"
                             "ExtractorResult. Set the invalid_results"
                             "parameter to 'ignore' if you wish to ignore"
                             "this.")

    if len(dfs) == 0:
        return pd.DataFrame()

    data = pd.concat(dfs, axis=0).reset_index(drop=True)

    if object_id == 'auto' and data['object_id'].nunique() == 1:
        data = data.drop('object_id', axis=1)

    if extractor_names in ['prepend', 'multi']:
        data['feature'] = data['extractor'] + '#' + data['feature'].astype(str)

    if extractor_names != 'column':
        data = data.drop('extractor', axis=1)

    if format == 'wide':
        ind_cols = {'stim_name', 'onset', 'order', 'duration', 'object_id',
                    'class', 'filename', 'history', 'source_file'}
        ind_cols = list(ind_cols & set(data.columns))
        # pandas groupby/index operations can't handle NaNs in index, (see
        # issue at https://github.com/pandas-dev/pandas/issues/3729), so we
        # replace NaNs with a placeholder and then re-substitute after
        # pivoting.
        dtypes = data[ind_cols].dtypes
        data[ind_cols] = data[ind_cols].fillna('PlAcEholdER')

        # Set default aggfunc based on column type, otherwise bad things happen
        if aggfunc is None:
            aggfunc = 'mean' if is_numeric_dtype(data['value']) else 'first'

        data = data.pivot_table(index=ind_cols, columns='feature',
                                values='value', aggfunc=aggfunc).reset_index()
        data.columns.name = None  # vestigial--is set to 'feature'
        data[ind_cols] = data[ind_cols].replace('PlAcEholdER', np.nan)
        data[ind_cols] = data[ind_cols].astype(dict(zip(ind_cols, dtypes)))

    if timing == 'auto' and 'onset' in data.columns:
        if data['onset'].isnull().all():
            data = data.drop(['onset', 'order', 'duration'], axis=1)

    if 'onset' in data.columns:
        key = [('onset', ''), ('order', ''), ('duration', '')] \
            if isinstance(data.columns, pd.MultiIndex) \
            else ['onset', 'order', 'duration']
        data = data.sort_values(key).reset_index(drop=True)

    if extractor_names == 'multi':
        if format == 'long':
            raise ValueError("Invalid extractor_names value 'multi'. When "
                             "format is 'long', extractor_names must be "
                             "one of 'drop', 'prepend', or 'column'.")
        data.columns = pd.MultiIndex.from_tuples(
            [c.split('#') for c in data.columns])
    return data
