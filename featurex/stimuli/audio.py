from featurex.stimuli import Stim
from featurex.stimuli.text import ComplexTextStim
from featurex.extractors import ExtractorResult
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
        self._load_clip(self.filename)


class TranscribedAudioStim(AudioStim):

    ''' An AudioStim with an associated text transcription.
    Args:
        filename (str): The path to the audio clip.
        transcription (str or ComplexTextStim): the associated transcription.
            If a string, this is interpreted as the name of a file containing
            data needed to initialize a new ComplexTextStim. Otherwise, must
            pass an existing ComplexTextStim instance.
        kwargs (dict): optional keywords passed to the ComplexTextStim
            initializer if transcription argument is a string.
    '''

    def __init__(self, filename, transcription, onset=None, **kwargs):
        if isinstance(transcription, six.string_types):
            transcription = ComplexTextStim(transcription, **kwargs)
        self.transcription = transcription
        super(TranscribedAudioStim, self).__init__(filename, onset=onset)


