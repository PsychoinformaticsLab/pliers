from abc import ABCMeta, abstractmethod, abstractproperty


class Transformer(object):

    __metaclass__ = ABCMeta

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

    @abstractmethod
    def transform(self):
        pass

    @abstractproperty
    def __version__(self):
        pass