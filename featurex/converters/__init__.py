from abc import ABCMeta, abstractmethod, abstractproperty
from featurex.transformers import Transformer, CollectionStimMixin
from six import with_metaclass
import importlib

__all__ = ['api', 'audio', 'google', 'image', 'video']


class Converter(with_metaclass(ABCMeta, Transformer)):
    ''' Base class for Converters.'''

    def convert(self, stim, *args, **kwargs):
        new_stim = self._convert(stim, *args, **kwargs)
        if new_stim.name is None:
            new_stim.name = stim.name
        else:
            new_stim.name = stim.name + '_' + new_stim.name
        if isinstance(new_stim, CollectionStimMixin):
            for s in new_stim:
                if s.name is None:
                    s.name = stim.name
                else:
                    s.name = stim.name + '_' + s.name
        return new_stim

    @abstractmethod
    def _convert(self, stim):
        pass

    @abstractproperty
    def _output_type(self):
        pass

    def _transform(self, stim, *args, **kwargs):
        return self.convert(stim, *args, **kwargs)


class MultistepConverter(Converter):
    ''' Base class for Converters doing more than one step.
    Args:
        via (list): Ordered sequence of types to convert through
    '''

    def __init__(self, via=None):
        super(MultistepConverter, self).__init__()
        self.via = self._via if via is None else via

    def _convert(self, stim):
        for i, step in enumerate(self.via):
            converter = get_converter(type(stim), step)
            if converter:
                stim = converter.transform(stim)
            else:
                msg = "Conversion failed during step %d" % i
                raise ValueError(msg)
        return stim


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

    base = Converter

    bm = base.__module__
    submods = importlib.import_module(bm).__all__
    sources = ['%s.%s' % (bm, sm) for sm in submods]
    [importlib.import_module(s) for s in sources]

    transformers = get_subclasses(base)
    for a in transformers:
        concrete = len(a.__abstractmethods__) == 0
        if concrete and issubclass(in_type, a._input_type) and issubclass(out_type, a._output_type):
            try:
                conv = a()
                return conv
            except ValueError:
                # Important for API converters
                pass
    return None
