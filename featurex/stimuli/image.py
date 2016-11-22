from featurex.stimuli import Stim
from featurex.extractors import ExtractorResult
from scipy.misc import imread
import six


class ImageStim(Stim):

    ''' A static image. '''

    def __init__(self, filename=None, data=None, duration=None):
        if data is None and isinstance(filename, six.string_types):
            data = imread(filename)
        super(ImageStim, self).__init__(filename)
        self.data = data
        self.duration = duration

    def extract(self, extractors):
        vals = []
        for e in extractors:
            vals.append(e.transform(self))
        return ExtractorResult.merge_features(vals)
