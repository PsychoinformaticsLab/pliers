from abc import ABCMeta, abstractmethod, abstractproperty
from pliers.transformers import Transformer, CollectionStimMixin
from six import with_metaclass
from pliers.utils import memory
import importlib
from types import GeneratorType
from pliers import config
import pliers


class Converter(with_metaclass(ABCMeta, Transformer)):
    ''' Base class for Converters.'''

    def __init__(self):
        super(Converter, self).__init__()
        if config.cache_converters:
            self.transform = memory.cache(self.transform)

    def convert(self, stim, *args, **kwargs):
        return self.transform(stim, *args, **kwargs)

    @abstractmethod
    def _convert(self, stim):
        pass

    @abstractproperty
    def _output_type(self):
        pass

    def _transform(self, stim, *args, **kwargs):
        new_stim = self._convert(stim, *args, **kwargs)
        if isinstance(new_stim, (list, tuple, GeneratorType)):
            return new_stim
        if new_stim.name is None:
            new_stim.name = stim.name
        else:
            new_stim.name = stim.name + '->' + new_stim.name
        if isinstance(new_stim, CollectionStimMixin):
            for s in new_stim:
                if s.name is None:
                    s.name = stim.name
        return new_stim


def get_converter(in_type, out_type, *args, **kwargs):
    ''' Scans the list of available Converters and returns an instantiation
    of the first one whose input and output types match those passed in.
    Args:
        in_type (type): The type of input the converter must have.
        out_type (type): The type of output the converter must have.
        args, kwargs: Optional positional and keyword arguments to pass onto
            matching Converter's initializer.
    '''
    convs = pliers.converters.__all__

    for name in convs:
        cls = getattr(pliers.converters, name)
        concrete = len(cls.__abstractmethods__) == 0
        if cls._input_type == in_type and cls._output_type == out_type and concrete:
            try:
                conv = cls(*args, **kwargs)
                return conv
            except ValueError:
                # Important for API converters
                pass
    return None
