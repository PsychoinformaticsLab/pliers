from abc import ABCMeta, abstractmethod
from featurex.core import Transformer
from six import with_metaclass


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


class Extractor(with_metaclass(ABCMeta, Transformer)):

    ''' Base Extractor class. Defines a target Stim class that all subclasses
    must override. '''
    # @abstractmethod
    # def extract(self):
    #     pass
