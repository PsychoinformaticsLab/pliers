from abc import ABCMeta, abstractmethod
from featurex.core import Transformer
from six import with_metaclass


__all__ = []


class Filter(with_metaclass(ABCMeta, Transformer)):
    ''' Base class for Filters.'''

    def filter(self, stim, *args, **kwargs):
        return self._filter(stim, *args, **kwargs)

    @abstractmethod
    def _filter(self, stim):
        pass

    def _transform(self, stim, *args, **kwargs):
        return self.filter(stim, *args, **kwargs)