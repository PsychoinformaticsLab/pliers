from six import with_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty
from featurex.stimuli import CollectionStimMixin
from featurex.utils import listify
import importlib


class Transformer(with_metaclass(ABCMeta)):

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

    def transform(self, stims, *args, **kwargs):
        # Iterate over all the stims in the list
        if isinstance(stims, (list, tuple)):
            return self._iterate(stims, *args, **kwargs)
        # Iterate over the collection of stims contained in the input stim
        elif isinstance(stims, CollectionStimMixin) and \
           not issubclass(self._input_type, CollectionStimMixin):
            return self._iterate(list(s for s in stims))
        # Pass the stim directly to the Transformer
        else:
            return self._transform(self._validate(stims), *args, **kwargs)

    def _validate(self, stim):
        if not isinstance(stim, self._input_type):
            msg = "Transformers of type %s can only be applied to stimuli of "\
                      "type(s) %s, not type %s."
            msg = msg % (self.__class__.__name__, self._input_type.__name__,
                        stim.__class__.__name__)

            converter = get_converter(type(stim), self._input_type)
            if converter:
                stim = converter.transform(stim)
            else:
                raise TypeError(msg)
        return stim

    def _iterate(self, stims, *args, **kwargs):
        return [self._transform(self._validate(s), *args, **kwargs) for s in stims]

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

    from featurex.converters import Converter
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
        from featurex.extractors import Extractor
        from featurex.converters import Converter
        from featurex.filters import Filter
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
