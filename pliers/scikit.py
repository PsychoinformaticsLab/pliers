from pliers.extractors import Extractor, merge_results

try:
    from sklearn.base import TransformerMixin, BaseEstimator
except:
    TransformerMixin, BaseEstimator = None


class PliersTransformer(BaseEstimator, TransformerMixin):

    ''' Simple wrapper for using pliers within a sklearn workflow.
    Args:
        transformer (Graph or Transformer): Pliers object to execute. Can
            either be a Graph with several transformers chained or a single
            transformer.
    '''

    def __init__(self, transformer):
        self.transformer = transformer
        self.metadata = None

    def fit(self, X, y=None):
        return self

    def fit_transform(self, stimulus_files, y=None):
        return self.transform(stimulus_files)

    def transform(self, stimulus_files):
        if isinstance(self.transformer, Extractor):
            result = self.transformer.transform(stimulus_files).to_df()
        else:
            result = self.transformer.transform(stimulus_files, merge=False)
            result = merge_results(result, flatten_columns=True)

        extra_columns = list(set(['onset', 'duration', 'history', 'class',
                                  'filename', 'stim_name', 'source_file']) &
                             set(result.columns))
        self.metadata = result[extra_columns]
        result.drop(extra_columns, axis=1, inplace=True, errors='ignore')

        return result.as_matrix()
