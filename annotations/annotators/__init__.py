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
