import collections
from six import string_types
from tqdm import tqdm
from pliers import config
from types import GeneratorType


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
