from featurex.stimuli import DynamicStim
from featurex.core import Timeline
from featurex.stimuli.text import ComplexTextStim
from scipy.io import wavfile
import six


class AudioStim(DynamicStim):

    ''' An audio clip. For now, only handles wav files. '''

    def __init__(self, filename):
        self.filename = filename
        self.sampling_rate, self.data = wavfile.read(filename)
        self._extract_duration()
        super(AudioStim, self).__init__(filename)

    def _extract_duration(self):
        self.duration = len(self.data)*1./self.sampling_rate

    def extract(self, extractors, merge_events=True):
        timeline = Timeline()
        for ext in extractors:
            events = ext.apply(self)
            for ev in events:
                timeline.add_event(ev, merge=merge_events)
        return timeline


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

    def __init__(self, filename, transcription, **kwargs):
        if isinstance(transcription, six.string_types):
            transcription = ComplexTextStim(transcription, **kwargs)
        self.transcription = transcription
        super(AudioStim, self).__init__(filename)

    def extract(self, extractors):
        timeline = Timeline()
        audio_exts, text_exts = [], []
        for ext in extractors:
            if ext.target.__name__ == 'AudioStim':
                audio_exts.append(ext)
            elif ext.target.__name__ == 'ComplexTextStim':
                text_exts.append(ext)

        audio_tl = super(TranscribedAudioStim, self).extract(audio_exts)
        timeline.merge(audio_tl)
        text_tl = self.transcription.extract(text_exts)
        timeline.merge(text_tl)
        return timeline
