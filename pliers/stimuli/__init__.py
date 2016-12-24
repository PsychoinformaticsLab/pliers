from abc import ABCMeta, abstractmethod, abstractproperty
from six import string_types
from os.path import exists, isdir, join, basename
from glob import glob
from six import with_metaclass, string_types
from pliers.utils import listify
import importlib


__all__ = ['audio', 'image', 'text', 'video']


class Stim(with_metaclass(ABCMeta)):

    ''' Base Stim class. '''
    def __init__(self, filename=None, onset=None, duration=None):

        self.filename = filename
        self.name = basename(filename) if filename is not None else str(self.id)
        self.features = []
        self.onset = onset
        self.duration = duration


    @property
    def id(self):
        return ''


class CollectionStimMixin(with_metaclass(ABCMeta)):

    @abstractmethod
    def __iter__(self):
        pass


class CompoundStim(object):

    ''' A container for an arbitrary set of Stims.
    Args:
        stims (Stim or list): a single Stim (of any type) or a list of Stims.

    '''
    def __init__(self, stims):

        self.stims = listify(stims)

    def get_stim(self, type_, return_all=False):
        ''' Returns component Stims of the specified type.
        Args:
            type_ (str or Stim class): the desired Stim subclass to return.
            return_all (bool): when True, returns all stims that matched the
                specified type as a list. When False (default), returns only
                the first matching Stim.
        Returns:
            If return_all is True, a list of matching Stims (or an empty list
            if no Stims match). If return_all is False, returns the first
            matching Stim, or None if no Stims match.
        '''
        if isinstance(type_, string_types):
            type_ = _get_stim_class(type_)
        matches = []
        for s in self.stims:
            if isinstance(s, type_):
                if not return_all:
                    return s
                matches.append(s)
        if not matches:
            return [] if return_all else None
        return matches

    def __getattr__(self, attr):
        try:
            stim = _get_stim_class(attr)
        except:
            raise AttributeError()
        return self.get_stim(stim)


def _get_stim_class(name):

    name = name.lower().replace('_', '')

    if not name.endswith('stim'):
        name += 'stim'

    # Recursively get all classes that inherit from the passed base class
    def get_subclasses(cls):
        subclasses = []
        for sc in cls.__subclasses__():
            subclasses.append(sc)
            subclasses.extend(get_subclasses(sc))
        return subclasses

    # Import all submodules so we have a comprehensive list of Stims
    base = Stim
    bm = base.__module__
    submods = importlib.import_module(bm).__all__
    sources = ['%s.%s' % (bm, sm) for sm in submods]
    [importlib.import_module(s) for s in sources]

    stims = get_subclasses(base)
    for a in stims:
        if a.__name__.lower().split('.')[-1] == name.lower():
            return a

    raise KeyError("No Stim class matches '%s' (case-insensitive)." % name)


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
