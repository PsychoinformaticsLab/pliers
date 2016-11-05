from os.path import dirname, join
from featurex.extractors import Extractor, ExtractorResult
from featurex.stimuli.image import ImageStim


def get_test_data_path():
    """Returns the path to test datasets """
    return join(dirname(__file__), 'data')


class DummyExtractor(Extractor):
    ''' A dummy Extractor class that always returns random values when
    extract() is called. Can set the extractor name inside _extract() to
    facilitate testing of results merging etc. '''
    target = ImageStim

    def _extract(self, stim, name=None, n_rows=100, n_cols=3, max_time=1000):
        data = np.random.randint(0, 1000, (n_rows, n_cols))
        onsets = np.random.choice(n_rows*2, n_rows, False)
        if name is not None:
            self.name = name
        return ExtractorResult(data, stim, deepcopy(self), onsets=onsets)