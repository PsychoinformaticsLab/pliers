''' Classes that represent audio clips. '''

from .base import Stim
from moviepy.audio.io.AudioFileClip import AudioFileClip

import os
import re
import subprocess


class AudioStim(Stim):

    ''' Represents an audio clip.

    Args:
        filename (str): Path to audio file.
        onset (float): Optional onset of the audio file (in seconds) with
            respect to some more general context or timeline the user wishes
            to keep track of.
        sampling_rate (int): Sampling rate of clip, in hertz.
        url (str): Optional url to read contents from.
        clip (AudioFileClip): Optional moviepy AudioFileClip to initialize
            from.
        order (int): Optional sequential index of the AudioStim within some
            containing context.

    '''

    _default_file_extension = '.wav'

    def __init__(self, filename=None, onset=None, sampling_rate=None, url=None,
                 clip=None, order=None):
        if url is not None:
            filename = url
        self.filename = filename

        self.sampling_rate = sampling_rate
        if not self.sampling_rate:
            self.sampling_rate = self.get_sampling_rate(self.filename)

        self.clip = clip
        if not self.clip:
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
            filename, onset=onset, duration=duration, order=order)

    @staticmethod
    def get_sampling_rate(filename):
        ''' Use FFMPEG to get the sampling rate, most of this code was
        adapted from the moviepy codebase '''
        cmd = ['ffmpeg', '-i', filename]

        with open(os.devnull, 'rb') as devnull:
            creationflags = 0x08000000 if os.name == 'nt' else 0
            p = subprocess.Popen(cmd,
                                 stdin=devnull,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 creationflags=creationflags)

        _, p_err = p.communicate()
        del p

        lines = p_err.decode('utf8').splitlines()
        if 'No such file or directory' in lines[-1]:
            raise IOError(('Error: the file %s could not be found.\n'
                           'Please check that you entered the correct '
                           'path.') % filename)

        lines_audio = [l for l in lines if ' Audio: ' in l]

        if lines_audio:
            line = lines_audio[0]
            try:
                match = re.search(' [0-9]* Hz', line)
                return int(line[match.start() + 1:match.end() - 3])
            except Exception as e:
                pass

        # Return a sensible default
        return 44100

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
        ''' Save clip data to file.

        Args:
            path (str): Filename to save audio data to.
        '''
        self.clip.write_audiofile(path, fps=self.sampling_rate)
