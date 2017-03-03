''' Classes that represent remote stimuli. '''

from .base import Stim
from .video import VideoStim, ImageStim
from .audio import AudioStim
from .text import TextStim

from six.moves.urllib.request import urlopen
from PIL import Image
import io
import numpy as np


class RemoteStim(Stim):

    ''' Represents a remote Stim accessible via URL.
    Args:
        url (str): URL to input.
    '''

    def __init__(self, url):
        self.url = url
        main_type = urlopen(url).info().getmaintype()
        stim_map = {
            'image': ImageStim,
            'video': VideoStim,
            'text': TextStim,
            'audio': AudioStim
        }
        if main_type in stim_map.keys():
            self.content_type = stim_map[main_type]

    def convert(self):
        if self.content_type is VideoStim or self.content_type is AudioStim:
            return self.content_type(self.url)
        elif self.content_type is ImageStim:
            img = Image.open(io.BytesIO(urlopen(self.url).read()))
            return ImageStim(data=np.array(img))
        else:
            return TextStim(text=urlopen(self.url).read())
