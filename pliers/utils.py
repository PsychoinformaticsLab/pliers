import collections
from six import string_types
from joblib import Memory
from tempfile import mkdtemp

cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=0)


def listify(obj):
    ''' Wraps all non-list or tuple objects in a list; provides a simple way
    to accept flexible arguments. '''
    return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def flatten(l):
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