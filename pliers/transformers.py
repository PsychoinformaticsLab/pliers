from pliers.stimuli.base import (Stim, CollectionStimMixin,
                                 _log_transformation, load_stims)
from pliers.stimuli.compound import CompoundStim
from pliers.utils import listify
from pliers import config
import pliers

from six import with_metaclass, string_types
from abc import ABCMeta, abstractmethod, abstractproperty
import importlib
from copy import deepcopy
import pandas as pd
from types import GeneratorType


class Transformer(with_metaclass(ABCMeta)):

    _log_attributes = ()

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

    def transform(self, stims, *args, **kwargs):

        if isinstance(stims, string_types):
            stims = load_stims(stims)

        # If stims is a CompoundStim and the Transformer is expecting a single
        # input type, extract all matching stims
        if isinstance(stims, CompoundStim) and not isinstance(self._input_type, tuple):
            stims = stims.get_stim(self._input_type, return_all=True)
            if not stims:
                raise ValueError("No stims of class %s found in the provided"
                                 "CompoundStim instance." % self._input_type)

        # If stims is an iterable, naively loop over elements.
        if isinstance(stims, (list, tuple, GeneratorType)):
            return self._iterate(stims, *args, **kwargs)

        # Validate stim, and then either pass it directly to the Transformer
        # or, if a conversion occurred, recurse.
        else:
            validated_stim = self._validate(stims)
            # If a conversion occurred during validation, we recurse
            if stims is not validated_stim:
                return self.transform(validated_stim, *args, **kwargs)
            else:
                result = self._transform(self._validate(stims), *args, **kwargs)
                result = _log_transformation(stims, result, self)
                if isinstance(result, GeneratorType):
                    result = list(result)
                return result

    def _validate(self, stim):
        if not isinstance(stim, self._input_type) and not \
               (isinstance(stim, CompoundStim) and stim.has_types(self._input_type)):
            from pliers.converters.base import get_converter
            converter = get_converter(type(stim), self._input_type)
            if converter:
                _old_stim = stim
                stim = converter.transform(stim)
                stim = _log_transformation(_old_stim, stim, converter)
            else:
                msg = "Transformers of type %s can only be applied to stimuli " \
                      " of type(s) %s (not type %s), and no applicable " \
                      "Converter was found."
                msg = msg % (self.__class__.__name__, self._input_type.__name__,
                        stim.__class__.__name__)
                raise TypeError(msg)
        return stim

    def _iterate(self, stims, *args, **kwargs):
        return (self.transform(s, *args, **kwargs) for s in stims)

    @abstractmethod
    def _transform(self, stim):
        pass

    @abstractproperty
    def _input_type(self):
        pass


class BatchTransformerMixin():
    ''' A mixin that overrides the default implicit iteration behavior. Use
    whenever batch processing of multiple stimuli should be handled within the
    _transform method rather than applying a naive loop--e.g., for API
    Extractors that can handle list inputs. '''
    def transform(self, stims, *args, **kwargs):
        return self._transform(self._validate(stims), *args, **kwargs)


def get_transformer(name, base=None, *args, **kwargs):
    ''' Scans list of currently available Transformer classes and returns an
    instantiation of the first one whose name perfectly matches
    (case-insensitive).
    Args:
        name (str): The name of the transformer to retrieve. Case-insensitive;
            e.g., 'stftextractor' or 'CornerDetectionExtractor'.
        base (str, list): Optional name of transformer modules to search.
            Valid values are 'converters', 'extractors', and 'filters'.
        args, kwargs: Optional positional or keyword arguments to pass onto
            the Transformer.
    '''

    name = name.lower()

    # Default to searching all kinds of Transformers
    if base is None:
        base = ['extractors', 'converters', 'filters']

    base = listify(base)

    for b in base:
        importlib.import_module('pliers.%s' % b)
        mod = getattr(pliers, b)
        classes = getattr(mod, '__all__')
        for cls_name in classes:
            if cls_name.lower() == name.lower():
                cls = getattr(mod, cls_name)
                return cls(*args, **kwargs)

    raise KeyError("No transformer named '%s' found." % name)
