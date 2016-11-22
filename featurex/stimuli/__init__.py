from abc import ABCMeta, abstractmethod, abstractproperty
from six import string_types
from os.path import exists, isdir, join, basename
from glob import glob
from six import with_metaclass


class Stim(with_metaclass(ABCMeta)):

    ''' Base Stim class. '''
    def __init__(self, filename=None, onset=None, duration=None):

        self.filename = filename
        self.features = []
        self.onset = onset
        self.duration = duration

    @property
    def name(self):
        filename = self.filename
        return basename(filename)if filename is not None else self.id


class CollectionStimMixin(with_metaclass(ABCMeta)):

    @abstractmethod
    def __iter__(self):
        pass


def load_stims(source, dtype=None):
    """ Load one or more stimuli directly from file, inferring/extracting
    metadata as needed.
    Args:
        source (str or list): The location to load the stim(s) from. Can be
            the path to a directory, to a single file, or a list of filenames.
        dtype (str): The type of stim to load. If dtype is None, relies on the
            filename extension for guidance. If dtype is provided, must be
            one of 'video', 'image', 'audio', or 'text'.
    Returns: A list of Stims.
    """
    import magic
    from .video import VideoStim, ImageStim
    from .audio import AudioStim
    from .text import TextStim

    if isinstance(source, string_types):
        source = [source]

    source = [s for s in source if exists(s)]

    stims = []

    def load_file(source):
        mime = magic.from_file(source, mime=True)
        if not isinstance(mime, string_types):
            mime = mime.decode('utf-8')
        mime = mime.split('/')[0]
        stim_map = {
            'image': ImageStim,
            'video': VideoStim,
            'text': TextStim,
            'audio': AudioStim
        }
        if mime in stim_map.keys():
            s = stim_map[mime](source)
            stims.append(s)

    for s in source:
        if isdir(s):
            for f in glob(join(s, '*')):
                load_file(f)
        else:
            load_file(s)

    return stims
