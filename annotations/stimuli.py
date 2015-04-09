from abc import ABCMeta, abstractmethod
import subprocess
import re


class Stim(object):

    __metaclass__ = ABCMeta

    def __init__(self, filename, label, description):

        self.filename = filename
        self.label = label
        self.description = description
        self.annotations = []


class DynamicStim(Stim):

    __metaclass__ = ABCMeta

    def __init__(self, filename, label, description):
        """ Any Stim object with a temporal dimension. """
        super(DynamicStim, self).__init__(filename, label, description)
        self._extract_duration()

    @abstractmethod
    def _extract_duration(self):
        pass

    @classmethod
    def ffprobe_duration(self):
        result = subprocess.Popen(
            ["ffprobe", self.filename], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        result = [x for x in result.stdout.readlines() if "Duration" in x]
        self.duration = re.search('Duration: (.*?),', result[0]).group(1)


class ImageStim(Stim):
    pass


class VideoStim(DynamicStim):

    def __init__(self, filename, label=None, description=None):
        super(VideoStim, self).__init__(filename, label, description)

    def _extract_duration(self):
        DynamicStim.ffprobe_duration()


class AudioStim(DynamicStim):

    def __init__(self, filename, label=None, description=None):
        super(VideoStim, self).__init__(filename, label, description)

    def _extract_duration(self):
        DynamicStim.ffprobe_duration()


class TextStim(DynamicStim):
    pass


class StimCollection(object):
    pass
