from pliers.transformers import Transformer
from pliers import config
from abc import ABCMeta, abstractmethod
from six import with_metaclass


class Filter(with_metaclass(ABCMeta, Transformer)):
    ''' Base class for Filters.'''

    def __init__(self):
        super(Filter, self).__init__()

    def transform(self, stim, *args, **kwargs):
        new_stim = self._filter(stim, *args, **kwargs)
        if not isinstance(new_stim, stim.__class__):
            raise ValueError("Filter must return a Stim of the same type as "
                             "its input.")
        return new_stim

    @abstractmethod
    def _filter(self, stim):
        pass

    def _transform(self, stim, *args, **kwargs):
        return self.filter(stim, *args, **kwargs)
