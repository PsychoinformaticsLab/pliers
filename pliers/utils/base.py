''' Miscellaneous internal utilities. '''

import collections
import os
from abc import ABCMeta, abstractmethod, abstractproperty
from six import string_types, with_metaclass
from tqdm import tqdm
from pliers import config
from pliers.support.exceptions import MissingDependencyError
from types import GeneratorType
from itertools import islice


def listify(obj):
    ''' Wraps all non-list or tuple objects in a list; provides a simple way
    to accept flexible arguments. '''
    return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def flatten(l):
    ''' Flatten an iterable. '''
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, string_types):
            for sub in flatten(el):
                yield sub
        else:
            yield el


def flatten_dict(d, parent_key='', sep='_'):
    ''' Flattens a multi-level dictionary into a single level by concatenating
    nested keys with the char provided in the sep argument. 

    Solution from https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys'''
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def batch_iterable(l, n):
    ''' Chunks iterable into n sized batches
    Solution from: http://stackoverflow.com/questions/1915170/split-a-generator-iterable-every-n-items-in-python-splitevery'''
    i = iter(l)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


def set_iterable_type(obj):
    ''' Returns either a generator or a list depending on config-level
    settings. Should be used to wrap almost every internal iterable return.
    Also inspects elements recursively in the case of list returns, to
    ensure that there are no nested generators. '''
    if not isiterable(obj):
        return obj

    if config.get_option('use_generators'):
        return obj if isgenerator(obj) else (i for i in obj)
    else:
        return [set_iterable_type(i) for i in obj]


class classproperty(object):
    ''' Implements a @classproperty decorator analogous to @classmethod.
    Solution from: http://stackoverflow.com/questions/128573/using-property-on-classmethodss
    '''
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def isiterable(obj):
    ''' Returns True if the object is one of allowable iterable types. '''
    return isinstance(obj, (list, tuple, GeneratorType, tqdm))


def isgenerator(obj):
    ''' Returns True if object is a generator, or a generator wrapped by a
    tqdm object. '''
    return isinstance(obj, GeneratorType) or (hasattr(obj, 'iterable') and
           isinstance(getattr(obj, 'iterable'), GeneratorType))


def progress_bar_wrapper(iterable, **kwargs):
    ''' Wrapper that applies tqdm progress bar conditional on config settings.
    '''
    return tqdm(iterable, **kwargs) if (config.get_option('progress_bar')
        and not isinstance(iterable, tqdm)) else iterable


module_names = {}
Dependency = collections.namedtuple('Dependency', 'package value')


def attempt_to_import(dependency, name=None, fromlist=None):
    if name is None:
        name = dependency
    try:
        mod = __import__(dependency, fromlist=fromlist)
    except ImportError:
        mod = None
    module_names[name] = Dependency(dependency, mod)
    return mod


def verify_dependencies(dependencies):
    missing = []
    for dep in listify(dependencies):
        if module_names[dep].value is None:
            missing.append(module_names[dep].package)
    if missing:
        raise MissingDependencyError(missing)


class EnvironmentKeyMixin(object):

    @classproperty
    def _env_keys(cls):
        pass

    @classproperty
    def env_keys(cls):
        return listify(cls._env_keys)

    @classproperty
    def available(cls):
        return all([k in os.environ for k in cls.env_keys])


class APIDependent(with_metaclass(ABCMeta, EnvironmentKeyMixin)):

    _rate_limit = 0

    def __init__(self, rate_limit=None, **kwargs):
        self.transformed_stim_count = 0
        self.validated_keys = set()
        self.rate_limit = rate_limit if rate_limit else self._rate_limit
        self._last_request_time = 0
        super(APIDependent, self).__init__(**kwargs)

    @abstractproperty
    def api_keys(self):
        pass

    def validate_keys(self):
        if all(k in self.validated_keys for k in self.api_keys):
            return True
        else:
            valid = self.check_valid_keys()
            if valid:
                for k in self.api_keys:
                    self.validated_keys.add(k)
            return valid

    @abstractmethod
    def check_valid_keys(self):
        pass
