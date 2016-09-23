from abc import ABCMeta, abstractmethod
from featurex.core import Transformer
from six import with_metaclass


__all__ = ['api', 'audio', 'google', 'image', 'text', 'video']


class Extractor(with_metaclass(ABCMeta, Transformer)):
    ''' Base class for Converters.'''

    def extract(self, stim, *args, **kwargs):
        return self._extract(stim, *args, **kwargs)

    @abstractmethod
    def _extract(self, stim):
        pass

    def _transform(self, stim, *args, **kwargs):
        return self.extract(stim, *args, **kwargs)
