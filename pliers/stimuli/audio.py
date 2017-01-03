from .base import Stim
from moviepy.audio.io.AudioFileClip import AudioFileClip
import six


class AudioStim(Stim):

    ''' An audio clip. For now, only handles wav files. '''

    def __init__(self, filename, onset=None, sampling_rate=44100):

        self.filename = filename
        self.sampling_rate = sampling_rate

        self._load_clip()

        self.data = self.clip.to_soundarray()
        duration = self.clip.duration

        if self.data.ndim > 1:
            # Average channels to make data mono
            self.data = self.data.mean(axis=1)

        super(AudioStim, self).__init__(filename, onset=onset, duration=duration)

    def _load_clip(self):
        self.clip = AudioFileClip(self.filename, fps=self.sampling_rate)

    def __getstate__(self):
        d = self.__dict__.copy()
        d['clip'] = None
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._load_clip()
