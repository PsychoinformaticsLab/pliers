from abc import ABCMeta, abstractproperty, abstractmethod

def strict(func):
    def wrapper(*args, **kwargs):
        cls, stim = args[:2]
        if not isinstance(stim, cls.target):
            stuff = (cls.__class__.__name__, stim.__class__.__name__)
            msg = "Extractors of type %s can only be applied to stimuli of type(s) %s, not %s."
            msg = msg % (cls.__class__.__name__, cls.target.__name__, stim.__class__.__name__)
            raise TypeError(msg)
        return func(*args, **kwargs)
    return wrapper


class Extractor(object):

    ''' Base Extractor class. '''

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self):
        pass


class StimExtractor(Extractor):

    ''' Abstract class for stimulus-specific extractors. Defines a target Stim
    class that all subclasses must override. '''
    @abstractproperty
    def target(self):
        pass


class ExtractorCollection(StimExtractor):

    def __init__(self, extractors=None):
        if extractors is None:
            extractors = []
        self.extractors = [get_extractor(s) for s in extractors]
        super(StimExtractor, self).__init__()

    def apply(self, stim, *args, **kwargs):
        return stim.extract(self.extractors, *args, **kwargs)


def get_extractor(name):
    ''' Scans list of currently available Extractor classes and returns an
    instantiation of the first one whose name perfectly matches
    (case-insensitive).
    Args:
        name (str): The name of the extractor to retrieve. Case-insensitive;
            e.g., 'stftextractor' or 'CornerDetectionExtractor'. For
            convenience, the 'extractor' suffix can be dropped--i.e., passing
            'stft' is equivalent to passing 'stftextractor'.
    '''

    name = name.lower()
    if not (name.endswith('extractor') or name.endswith('collection')):
        name += 'extractor'

    # Import all submodules so we have a comprehensive list of extractors
    from featurex.extractors import audio
    from featurex.extractors import image
    from featurex.extractors import text
    from featurex.extractors import video

    # Recursively get all classes that inherit from Extractor
    def get_subclasses(cls):
        subclasses = []
        for sc in cls.__subclasses__():
            subclasses.append(sc)
            subclasses.extend(get_subclasses(sc))
        return subclasses

    extractors = get_subclasses(Extractor)
    for a in extractors:
        if a.__name__.lower().split('.')[-1] == name.lower():
            return a()

    raise KeyError("No extractor named '%s' found." % name)
