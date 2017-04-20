from sklearn.base import TransformerMixin


class PliersTransformer(TransformerMixin):

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
        # TODO: add more postprocessing of ExtractorResult
        return self.transformer.transform(stimulus_files)
