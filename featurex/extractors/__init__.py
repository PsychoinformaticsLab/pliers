from abc import ABCMeta, abstractmethod
from featurex.core import Transformer
from six import with_metaclass


class Extractor(with_metaclass(ABCMeta, Transformer)):

    ''' Base Extractor class. Defines a target Stim class that all subclasses
    must override. '''
    # @abstractmethod
    # def extract(self):
    #     pass
