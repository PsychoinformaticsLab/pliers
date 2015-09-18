from abc import ABCMeta, abstractproperty, abstractmethod


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

    @abstractproperty
    def target(self):
        pass


def get_extractor(name):
    ''' Scans list of currently available Extractor classes and returns an
    instantiation of the first one whose name perfectly matches
    (case-insensitive).
    Args:
        name (str): The name of the extractor to retrieve. Case-insensitive;
            e.g., 'stftextractor' or 'CornerDetectionExtractor'.
    '''

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
