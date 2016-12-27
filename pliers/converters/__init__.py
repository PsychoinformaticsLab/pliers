from abc import ABCMeta, abstractmethod, abstractproperty
from pliers.transformers import Transformer, CollectionStimMixin
from six import with_metaclass
from pliers.utils import memory
import importlib


__all__ = ['api', 'audio', 'google', 'image', 'video', 'multistep']


class Converter(with_metaclass(ABCMeta, Transformer)):
    ''' Base class for Converters.'''

    def __init__(self):
        super(Converter, self).__init__()
        self.convert = memory.cache(self.convert)

    def convert(self, stim, *args, **kwargs):
        new_stim = self._convert(stim, *args, **kwargs)
        if new_stim.name is None:
            new_stim.name = stim.name
        else:
            new_stim.name = stim.name + '->' + new_stim.name
        if isinstance(new_stim, CollectionStimMixin):
            for s in new_stim:
                if s.name is None:
                    s.name = stim.name
        return new_stim

    @abstractmethod
    def _convert(self, stim):
        pass

    @abstractproperty
    def _output_type(self):
        pass

    def _transform(self, stim, *args, **kwargs):
        return self.convert(stim, *args, **kwargs)


def get_converter(in_type, out_type):
    ''' Scans the list of available Converters and returns an instantiation
    of the first one whose input and output types match those passed in.
    Args:
        in_type (type): The type of input the converter must have.
        out_type (type): The type of output the converter must have.
    '''

    # Recursively get all classes that inherit from the passed base class
    def get_subclasses(cls):
        subclasses = []
        for sc in cls.__subclasses__():
            subclasses.append(sc)
            subclasses.extend(get_subclasses(sc))
        return subclasses

    from pliers.converters import Converter
    base = Converter

    bm = base.__module__
    submods = importlib.import_module(bm).__all__
    sources = ['%s.%s' % (bm, sm) for sm in submods]
    [importlib.import_module(s) for s in sources]

    transformers = get_subclasses(base)
    for a in transformers:
        concrete = len(a.__abstractmethods__) == 0
        if a._input_type == in_type and a._output_type == out_type and concrete:
            try:
                conv = a()
                return conv
            except ValueError:
                # Important for API converters
                pass
    return None
