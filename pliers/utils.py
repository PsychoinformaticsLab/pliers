import collections
from abc import abstractproperty
from six import string_types
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


def batch_iterable(l, n):
    ''' Chunks iterable into n sized batches
    Solution from: http://stackoverflow.com/questions/1915170/split-a-generator-iterable-every-n-items-in-python-splitevery'''
    i = iter(l)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


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
    return tqdm(iterable, **kwargs) if (config.progress_bar and
        not isinstance(iterable, tqdm)) else iterable


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
    for dep in dependencies:
        if module_names[dep].value is None:
            missing.append(module_names[dep].package)
    if missing:
        raise MissingDependencyError(missing)


class EnvironmentKeyMixin(object):

    @abstractproperty
    def _env_keys(self):
        pass

    @property
    def env_keys(self):
        return listify(self._env_keys)

    @classproperty
    def available(cls):
        return True if all([k in os.environ for k in self.env_keys]) else False
