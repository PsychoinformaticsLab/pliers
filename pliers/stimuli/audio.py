''' Classes that represent audio clips. '''

import sndhdr

from .base import Stim
from moviepy.audio.io.AudioFileClip import AudioFileClip


class AudioStim(Stim):

    ''' Represents an audio clip.
    Args:
        filename (str): Path to audio file.
        onset (float): Optional onset of the audio file (in seconds) with
            respect to some more general context or timeline the user wishes
            to keep track of.
        sampling_rate (int): Sampling rate of clip, in hertz.

    '''

    _default_file_extension = '.wav'

    def __init__(self, filename=None, onset=None, sampling_rate=None, url=None, clip=None):
        if url is not None:
            filename = url
        self.filename = filename
        if sampling_rate:
            self.sampling_rate = sampling_rate
        else:
            self.sampling_rate = sndhdr.what(self.filename)[1]
        self.clip = clip

        if self.clip is None:
            self._load_clip()

        # Small default buffer isn't ideal, but moviepy has persistent issues
        # with some files otherwise; see
        # https://github.com/Zulko/moviepy/issues/246
        self.data = self.clip.to_soundarray(buffersize=1000)
        duration = self.clip.duration

        if self.data.ndim > 1:
            # Average channels to make data mono
            self.data = self.data.mean(axis=1)

        super(AudioStim, self).__init__(
            filename, onset=onset, duration=duration)

    def _load_clip(self):
        self.clip = AudioFileClip(self.filename, fps=self.sampling_rate)

    def __getstate__(self):
        d = self.__dict__.copy()
        d['clip'] = None
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._load_clip()

    def save(self, path):
        self.clip.write_audiofile(path)
