from abc import ABCMeta, abstractmethod


class Stim(object):

    ''' Base Stim class. '''

    __metaclass__ = ABCMeta

    def __init__(self, filename=None):

        self.filename = filename
        self.features = []


class DynamicStim(Stim):

    ''' Any Stim that has as a temporal dimension. '''

    __metaclass__ = ABCMeta

    def __init__(self, filename=None):
        super(DynamicStim, self).__init__(filename)
        self._extract_duration()

    @abstractmethod
    def _extract_duration(self):
        pass
