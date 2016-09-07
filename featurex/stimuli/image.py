from featurex.stimuli import Stim
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
