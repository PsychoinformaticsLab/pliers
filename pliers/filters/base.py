''' Base Filter class and associated functionality. '''

from abc import ABCMeta, abstractmethod
from six import with_metaclass
from pliers.transformers import Transformer


class Filter(with_metaclass(ABCMeta, Transformer)):
    ''' Base class for Filters.'''

    def __init__(self):
        super(Filter, self).__init__()

    def _transform(self, stim, *args, **kwargs):
        return self._filter(stim, *args, **kwargs)

    @abstractmethod
    def _filter(self, stim):
        pass
