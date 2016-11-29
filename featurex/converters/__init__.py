from abc import ABCMeta, abstractmethod, abstractproperty
from featurex.transformers import Transformer
from six import with_metaclass

__all__ = []


class Converter(with_metaclass(ABCMeta, Transformer)):
    ''' Base class for Converters.'''

    def convert(self, stim, *args, **kwargs):
        new_stim = self._convert(stim, *args, **kwargs)
        if new_stim.id is None:
            new_stim.name = stim.name
        else:
            new_stim.name = stim.name + '_' + new_stim.id
        return new_stim

    @abstractmethod
    def _convert(self, stim):
        pass

    @abstractproperty
    def _output_type(self):
        pass

    def _transform(self, stim, *args, **kwargs):
        return self.convert(stim, *args, **kwargs)

