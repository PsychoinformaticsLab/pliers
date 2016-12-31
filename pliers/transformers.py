from six import with_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty
from pliers.stimuli import Stim, CollectionStimMixin, _log_transformation
from pliers.utils import listify
import importlib
from copy import deepcopy
from pliers import config
import pandas as pd


class Transformer(with_metaclass(ABCMeta)):

    _log_attributes = ()

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

    def transform(self, stims, *args, **kwargs):

        # If stims is a plain list or tuple, just naively loop over elements.
        if isinstance(stims, (list, tuple)):
            return self._iterate(stims, *args, **kwargs)

        # If stims is a CompoundStim and the Transformer is expecting a single
        # input type, extract all matching stims
        from pliers.stimuli.compound import CompoundStim
        if isinstance(stims, CompoundStim) and not isinstance(self._input_type, tuple):
            stims = stims.get_stim(self._input_type, return_all=True)
            if not stims:
                raise ValueError("No stims of class %s found in the provided"
                                 "CompoundStim instance." % self._input_type)

        # If Stims can be iterated, and the Transformer is NOT expecting to be
        # handed an iterable Stim, then first try to convert the Stim to
        # something the Transformer can handle, and then fall back on naive
        # looping.
        elif isinstance(stims, CollectionStimMixin) and \
           not issubclass(self._input_type, CollectionStimMixin):
            from pliers.converters import get_converter
            converter = get_converter(type(stims), self._input_type)
            if converter:
                return self.transform(converter.transform(stims))
            else:
                return self._iterate(list(s for s in stims))

        # Otherwise pass the stim directly to the Transformer
        else:
            validated_stim = self._validate(stims)
            # If a conversion occurred during validation, we recurse--after
            # updating the history to reflect the conversion.
            if stims is not validated_stim:
                _log_transformation(stims, self, validated_stim)
                return self.transform(validated_stim, *args, **kwargs)
            else:
                result = self._transform(self._validate(stims), *args, **kwargs)
                _log_transformation(stims, self, result)
                return result

    def _validate(self, stim):
        from pliers.stimuli.compound import CompoundStim
        if not isinstance(stim, self._input_type) and not \
               (isinstance(stim, CompoundStim) and stim.has_types(self._input_type)):
            from pliers.converters import get_converter
            converter = get_converter(type(stim), self._input_type)
            if converter:
                stim = converter.transform(stim)
            else:
                msg = "Transformers of type %s can only be applied to stimuli " \
                      " of type(s) %s (not type %s), and no applicable " \
                      "Converter was found."
                msg = msg % (self.__class__.__name__, self._input_type.__name__,
                        stim.__class__.__name__)
                raise TypeError(msg)
        return stim

    def _iterate(self, stims, *args, **kwargs):
        return [self.transform(s, *args, **kwargs) for s in stims]

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
        args, kwargs: Optional positional or keyword arguments to pass onto
            the Transformer.
    '''

    name = name.lower()

    # Recursively get all classes that inherit from the passed base class
    def get_subclasses(cls):
        subclasses = []
        for sc in cls.__subclasses__():
            subclasses.append(sc)
            subclasses.extend(get_subclasses(sc))
        return subclasses

    # Default to searching all kinds of Transformers
    if base is None:
        from pliers.extractors import Extractor
        from pliers.converters import Converter
        from pliers.filters import Filter
        base = [Converter, Extractor, Filter]

    # Import all submodules so we have a comprehensive list of transformers
    base = listify(base)
    for b in base:
        bm = b.__module__
        submods = importlib.import_module(bm).__all__
        sources = ['%s.%s' % (bm, sm) for sm in submods]
        [importlib.import_module(s) for s in sources]

        transformers = get_subclasses(b)
        for a in transformers:
            if a.__name__.lower().split('.')[-1] == name.lower():
                return a(*args, **kwargs)

    raise KeyError("No transformer named '%s' found." % name)
