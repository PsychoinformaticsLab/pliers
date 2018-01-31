''' Contains sklearn-compatible wrappers for pliers. '''

from pliers.extractors import Extractor, merge_results
from pliers.transformers import get_transformer
from pliers.utils import attempt_to_import
from six import string_types

sklearn = attempt_to_import('sklearn')
if sklearn:
    class SklearnBase(sklearn.base.TransformerMixin,
                      sklearn.base.BaseEstimator):
        pass
else:
    class SklearnBase():
        pass


class PliersTransformer(SklearnBase):

    ''' Simple wrapper for using pliers within a sklearn workflow.

    Args:
        transformer (Graph or Transformer): Pliers object to execute. Can
            either be a Graph with several transformers chained or a single
            transformer.
    '''

    def __init__(self, transformer):
        if isinstance(transformer, string_types):
            self.transformer = get_transformer(transformer)
        else:
            self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, stimulus_files, y=None):
        return self.transform(stimulus_files)

    def transform(self, stimulus_files):
        if isinstance(self.transformer, Extractor):
            result = self.transformer.transform(stimulus_files).to_df()
        else:
            result = self.transformer.transform(stimulus_files, merge=False)
            result = merge_results(result, format='wide',
                                   extractor_names=False)

        extra_columns = list(set(['onset', 'duration', 'order', 'history',
                                  'class', 'filename', 'stim_name',
                                  'source_file', 'object_id', 'extractor']) &
                             set(result.columns))
        self.metadata_ = result[extra_columns]
        result.drop(extra_columns, axis=1, inplace=True, errors='ignore')

        return result.as_matrix()
