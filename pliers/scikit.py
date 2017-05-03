from pliers.extractors import ExtractorResult

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

    def fit(self, X, y=None):
        return self

    def fit_transform(self, stimulus_files, y=None):
        return self.transform(stimulus_files)

    def transform(self, stimulus_files):
        result = self.transformer.transform(stimulus_files)

        if isinstance(result, ExtractorResult):
            result = result.to_df()

        result.drop(['onset', 'duration', 'history', 'class',
                     'filename', 'stim_name', 'source_file'],
                    axis=1,
                    inplace=True,
                    errors='ignore')

        return result.as_matrix()
