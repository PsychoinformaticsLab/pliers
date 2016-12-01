from abc import ABCMeta, abstractmethod, abstractproperty
from featurex.transformers import Transformer
from six import with_metaclass

__all__ = ['api', 'audio', 'google', 'image', 'video']


class Converter(with_metaclass(ABCMeta, Transformer)):
    ''' Base class for Converters.'''

    def convert(self, stim, *args, **kwargs):
        return self._convert(stim, *args, **kwargs)

    @abstractmethod
    def _convert(self, stim):
        pass

    @abstractproperty
    def _output_type(self):
        pass

    def _transform(self, stim, *args, **kwargs):
        return self.convert(stim, *args, **kwargs)

