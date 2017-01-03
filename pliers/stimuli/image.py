from .base import Stim
from scipy.misc import imread
import six


class ImageStim(Stim):

    ''' A static image. '''

    def __init__(self, filename=None, onset=None, duration=None, data=None):
        if data is None and isinstance(filename, six.string_types):
            data = imread(filename)
        self.data = data
        super(ImageStim, self).__init__(filename, onset=onset, duration=duration)

