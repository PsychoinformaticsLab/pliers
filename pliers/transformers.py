''' Core transformer logic. '''

from pliers import config
from pliers.stimuli.base import Stim, _log_transformation, load_stims
from pliers.stimuli.compound import CompoundStim
from pliers.utils import (progress_bar_wrapper, isiterable,
                          isgenerator, listify, batch_iterable,
                          attempt_to_import)
import pliers
from six import with_metaclass, string_types
from abc import ABCMeta, abstractmethod, abstractproperty
import importlib
import logging

multiprocessing = attempt_to_import('pathos.multiprocessing',
                                    'multiprocessing', ['ProcessingPool'])

_cache = {}


class Transformer(with_metaclass(ABCMeta)):
    ''' Base class for all pliers Transformers. '''

    _log_attributes = ()
    _loggable = True
    VERSION = '0.1'

    # Stim types that *can* be passed as input, but aren't mandatory. This
    # allows for disjunctive specification; e.g., if _input_type is empty
    # and _optional_input_type is (AudioStim, TextStim), then _at least_ one
    # of the two must be passed. If both are specified in _input_type, then
    # the input would have to be a CompoundStim with both audio and text slots.
    _optional_input_type = ()

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

    def _memoize(transform):
        def wrapper(self, stim, *args, **kwargs):
            use_cache = config.cache_transformers and isinstance(stim, Stim)
            if use_cache:
                key = hash((hash(self), hash(stim)))
                if key in _cache:
                    return _cache[key]
            result = transform(self, stim, *args, **kwargs)
            if use_cache:
                if isgenerator(result):
                    result = list(result)
                _cache[key] = result
            return result
        return wrapper

    @_memoize
    def transform(self, stims, validation='strict', *args, **kwargs):
        ''' Executes the transformation on the passed stim(s).
        Args:
            stims (Stim, list): Stim or list of Stims to process.
            validation (str): String specifying how validation errors should
                be handled. Must be one of:
                    * 'strict': Raise an exception on any validation error
                    * 'warn': Issue a warning for all validation errors
                    * 'loose': Silently ignore all validation errors
            args: Optional positional arguments to pass onto the internal
                _transform call.
            kwargs: Optional positional arguments to pass onto the internal
                _transform call.
        '''

        if isinstance(stims, string_types):
            stims = load_stims(stims)

        # If stims is a CompoundStim and the Transformer is expecting a single
        # input type, extract all matching stims
        if isinstance(stims, CompoundStim) and not isinstance(self._input_type, tuple):
            stims = stims.get_stim(self._input_type, return_all=True)
            if not stims:
                raise ValueError("No stims of class %s found in the provided"
                                 "CompoundStim instance." % self._input_type)

        # If stims is an iterable, naively loop over elements, removing
        # invalid results if needed
        if isiterable(stims):
            iters = self._iterate(stims, *args, **kwargs)
            if config.drop_bad_extractor_results:
                iters = (i for i in iters if i is not None)
            return progress_bar_wrapper(iters, desc='Stim')

        # Validate stim, and then either pass it directly to the Transformer
        # or, if a conversion occurred, recurse.
        else:
            try:
                validated_stim = self._validate(stims)
            except TypeError as err:
                if validation == 'strict':
                    raise err
                elif validation == 'warn':
                    logging.warn(str(err))
                    return
                elif validation == 'loose':
                    return
            # If a conversion occurred during validation, we recurse
            if stims is not validated_stim:
                return self.transform(validated_stim, *args, **kwargs)
            else:
                result = self._transform(validated_stim, *args, **kwargs)
                result = _log_transformation(validated_stim, result, self)
                if isgenerator(result):
                    result = list(result)
                return result

    def _validate(self, stim):
        # Checks whether the current Transformer can handle the passed Stim.
        # If not, attempts a dynamic conversion before failing.
        if not self._stim_matches_input_types(stim):
            from pliers.converters.base import get_converter
            in_type = self._input_type if self._input_type \
                else self._optional_input_type
            converter = get_converter(type(stim), in_type)
            if converter:
                _old_stim = stim
                stim = converter.transform(stim)
                stim = _log_transformation(_old_stim, stim, converter)
            else:
                msg = ("Transformers of type %s can only be applied to stimuli"
                       " of type(s) %s (not type %s), and no applicable "
                       "Converter was found.")
                msg = msg % (self.__class__.__name__, in_type,
                             stim.__class__.__name__)
                raise TypeError(msg)
        return stim

    def _stim_matches_input_types(self, stim):
        # Checks if passed Stim meets all _input_type and _optional_input_type
        # specifications.

        mandatory = tuple(listify(self._input_type))
        optional = tuple(listify(self._optional_input_type))

        if isinstance(stim, CompoundStim):
            return stim.has_types(mandatory) or \
                (not mandatory and stim.has_types(optional, False))

        if len(mandatory) > 1:
            msg = "Transformer of class %s requires multiple mandatory " + \
                  "inputs, so the passed input Stim must be a CompoundStim" + \
                  "--which it isn't." % self.__class__.__name__
            raise ValueError(msg)

        return isinstance(stim, mandatory) or (not mandatory and
                                               isinstance(stim, optional))

    def _iterate(self, stims, *args, **kwargs):

        if config.parallelize and multiprocessing is not None:
            def _transform(s):
                return self.transform(s, *args, **kwargs)
            return multiprocessing.ProcessingPool(config.n_jobs) \
                .map(_transform, stims)

        return (t for t in (self.transform(s, *args, **kwargs)
                            for s in stims) if t)

    @abstractmethod
    def _transform(self, stim):
        pass

    @abstractproperty
    def _input_type(self):
        pass

    def __hash__(self):
        tr_attrs = [getattr(self, attr) for attr in self._log_attributes]
        return hash(self.name + str(dict(zip(self._log_attributes, tr_attrs))))


class BatchTransformerMixin(Transformer):
    ''' A mixin that overrides the default implicit iteration behavior. Use
    whenever batch processing of multiple stimuli should be handled within the
    _transform method rather than applying a naive loop--e.g., for API
    Extractors that can handle list inputs.
    Args:
        batch_size (int): Number of Stims to process in each batch.
        args, kwargs: Optional positional and keyword arguments to pass onto
            the base Transformer initializer.
    '''

    def __init__(self, batch_size=None, *args, **kwargs):
        if batch_size:
            self._batch_size = batch_size
        super(BatchTransformerMixin, self).__init__(*args, **kwargs)

    def _iterate(self, stims, *args, **kwargs):
        batches = batch_iterable(stims, self._batch_size)
        results = []
        for batch in batches:
            res = self._transform(batch, *args, **kwargs)
            for i, stim in enumerate(batch):
                res[i] = _log_transformation(stim, res[i], self)
            results.extend(res)
        return results

    def _transform(self, stim, *args, **kwargs):
        stims = listify(stim)
        if all(self._stim_matches_input_types(s) for s in stims):
            result = super(BatchTransformerMixin, self)._transform(stims, *args, **kwargs)
            if isiterable(stim):
                return result
            else:
                return result[0]
        else:
            return list(super(BatchTransformerMixin, self)._iterate(stims, *args, **kwargs))


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
