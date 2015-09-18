from abc import ABCMeta, abstractproperty, abstractmethod


class Annotator(object):

    ''' Base Annotator class. '''

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


def get_annotator(name):
    ''' Scans list of currently available Annotator classes and returns an
    instantiation of the first one whose name perfectly matches
    (case-insensitive).
    Args:
        name (str): The name of the annotator to retrieve. Case-insensitive;
            e.g., 'stftannotator' or 'CornerDetectionAnnotator'.
    '''

    # Recursively get all classes that inherit from Annotator
    def get_subclasses(cls):
        subclasses = []
        for sc in cls.__subclasses__():
            subclasses.append(sc)
            subclasses.extend(get_subclasses(sc))
        return subclasses

    annotators = get_subclasses(Annotator)

    for a in annotators:
        if a.__name__.lower().split('.')[-1] == name.lower():
            return a()
