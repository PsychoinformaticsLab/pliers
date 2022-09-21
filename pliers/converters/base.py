''' Base Converter class and utilities. '''

from abc import ABCMeta, abstractmethod, abstractproperty
import inspect

from pliers.transformers import Transformer
from pliers.utils import listify, EnvironmentKeyMixin
from pliers import config
import pliers


class Converter(Transformer, metaclass=ABCMeta):

    ''' Base class for Converters.'''

    @abstractmethod
    def _convert(self, stim):
        pass

    @abstractproperty
    def _output_type(self):
        pass

    def _transform(self, stim, *args, **kwargs):
        return self._convert(stim, *args, **kwargs)


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

    # If config includes default converters for this combination, try them 1st
    default_convs = config.get_option('default_converters')

    out_type = listify(out_type)[::-1]
    for ot in out_type:
        conv_str = '{}->{}'.format(in_type.__name__, ot.__name__)
        if conv_str in default_convs:
            convs = list(default_convs[conv_str]) + convs

    for name in convs:
        cls = getattr(pliers.converters, name)
        if not inspect.isclass(cls) or not issubclass(cls, Converter):
            continue

        # Some classes are only available if certain environment keys are set
        available = cls.available if issubclass(
            cls, EnvironmentKeyMixin) else True

        if cls._input_type == in_type and cls._output_type in out_type \
                and available:
            conv = cls(*args, **kwargs)
            return conv

    return None
