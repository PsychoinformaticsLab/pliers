from six import with_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty
from featurex.utils import listify
import importlib


class Transformer(with_metaclass(ABCMeta)):

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

    def transform(self, stim, *args, **kwargs):
        if not isinstance(stim, self.target):
            msg = "Transformers of type %s can only be applied to stimuli of "\
                  "type(s) %s, not type %s."
            msg = msg % (self.__class__.__name__, self.target.__name__,
                         stim.__class__.__name__)   
            raise TypeError(msg)      
        return self._transform(stim, *args, **kwargs)

    @abstractmethod
    def _transform(self, stim):
        pass

    @abstractproperty
    def target(self):
        pass

    # TODO: implement versioning on all subclasses
    # @abstractproperty
    # def __version__(self):
    #     pass


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
