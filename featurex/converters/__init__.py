from abc import ABCMeta, abstractmethod, abstractproperty
from featurex.transformers import Transformer, CollectionStimMixin
from six import with_metaclass
from tempfile import mkdtemp
from joblib import Memory

cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=0)

__all__ = ['api', 'audio', 'google', 'image', 'video']


class Converter(with_metaclass(ABCMeta, Transformer)):
    ''' Base class for Converters.'''

    def __init__(self):
        super(Converter, self).__init__()
        self.convert = memory.cache(self.convert)

    def convert(self, stim, *args, **kwargs):
        new_stim = self._convert(stim, *args, **kwargs)
        if new_stim.name is None:
            new_stim.name = stim.name
        else:
            new_stim.name = stim.name + '_' + new_stim.name
        if isinstance(new_stim, CollectionStimMixin):
            for s in new_stim:
                if s.name is None:
                    s.name = stim.name
                else:
                    s.name = stim.name + '_' + s.name
        return new_stim

    @abstractmethod
    def _convert(self, stim):
        pass

    @abstractproperty
    def _output_type(self):
        pass

    def _transform(self, stim, *args, **kwargs):
        return self.convert(stim, *args, **kwargs)
