''' Classes that represent audio clips. '''

from .base import Stim
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos


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

        if clip:
            self.sampling_rate = clip.fps
            self.clip = clip
        else:
            self.sampling_rate = sampling_rate
            if not self.sampling_rate:
                self.sampling_rate = self.get_sampling_rate(self.filename)
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
            filename, onset=onset, duration=duration, order=order, url=url)

    def _load_clip(self):
        self.clip = AudioFileClip(self.filename, fps=self.sampling_rate)

    @staticmethod
    def get_sampling_rate(filename):
        ''' Use moviepy/FFMPEG to get the sampling rate '''
        infos = ffmpeg_parse_infos(filename)
        fps = infos.get('audio_fps', 44100)
        if fps == 'unknown':
            fps = 44100
        return fps

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
