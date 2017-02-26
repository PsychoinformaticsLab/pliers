''' Classes that represent images. '''

from .base import Stim
from scipy.misc import imread
import six


class ImageStim(Stim):

    ''' Represents a static image.
    Args:
        filename (str): Path to input file, if one exists.
        onset (float): Optional onset of the image (in seconds) with
            respect to some more general context or timeline the user wishes
            to keep track of.
        duration (float): Optional duration of the ImageStim, in seconds.
        data (ndarray): Optional numpy array to initialize the image from,
            if no filename is passed.
    '''

    def __init__(self, filename=None, onset=None, duration=None, data=None):
        if data is None and isinstance(filename, six.string_types):
            data = imread(filename)
        self.data = data
        super(ImageStim, self).__init__(filename, onset=onset, duration=duration)
