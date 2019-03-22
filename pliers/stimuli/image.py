''' Classes that represent images. '''

from .base import Stim
from scipy.misc import imread
from PIL import Image
from six.moves.urllib.request import urlopen
from scipy.misc import imsave
import six
import io
import numpy as np
from functools import lru_cache
import base64


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
        url (str): Optional url to read contents from.
    '''

    _default_file_extension = '.png'

    def __init__(self, filename=None, onset=None, duration=None, data=None,
                 url=None):
        if data is None and isinstance(filename, six.string_types):
            data = imread(filename, mode='RGB')
        if url is not None:
            img = Image.open(io.BytesIO(urlopen(url).read()))
            img = img.convert(mode='RGB')
            data = np.array(img)
            filename = url
        self.data = data
        self._bytestring = None
        super(ImageStim, self).__init__(filename, onset=onset,
                                        duration=duration, url=url)

    def save(self, path):
        imsave(path, self.data)

    def __hash__(self):
        return hash((self.data.tobytes(), self.onset, self.duration,
                     self.order, self.history))
    

    def get_bytestring(self, encoding='utf-8'):
        ''' Return the image data as a bytestring.

        Args:
            encoding (str): Encoding to use. Defaults to utf-8.

        Returns: A string.
        '''
        if self._bytestring is None:
            with self.get_filename() as filename:
                with open(filename, 'rb') as f:
                    data = f.read()
                    self._bytestring = base64.b64encode(data).decode(encoding=encoding)

        return self._bytestring