from abc import ABCMeta, abstractmethod
import cv2
import six
from .core import Timeline, Event
import pandas as pd
from scipy.io import wavfile


class Stim(object):

    __metaclass__ = ABCMeta

    def __init__(self, filename=None):

        self.filename = filename
        self.annotations = []


class DynamicStim(Stim):

    ''' Any Stim that has as a temporal dimension. '''

    __metaclass__ = ABCMeta

    def __init__(self, filename=None):
        super(DynamicStim, self).__init__(filename)
        self._extract_duration()

    @abstractmethod
    def _extract_duration(self):
        pass


class ImageStim(Stim):

    ''' A static image. '''

    def __init__(self, filename=None, data=None, duration=None):
        if data is None and isinstance(filename, six.string_types):
            data = cv2.imread(filename)
        super(ImageStim, self).__init__(filename)
        self.data = data
        self.duration = duration


class VideoFrameStim(ImageStim):

    ''' A single frame of video. '''

    def __init__(self, video, frame_num, filename=None, data=None):
        super(VideoFrameStim, self).__init__(filename, data)
        self.video = video
        self.frame_num = frame_num
        self.duration = 1. / video.fps
        self.onset = frame_num * self.duration


class VideoStim(DynamicStim):

    ''' A video. '''

    def __init__(self, filename):
        self.clip = cv2.VideoCapture(filename)
        self.fps = self.clip.get(cv2.cv.CV_CAP_PROP_FPS)
        self.n_frames = self.clip.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        self.width = int(self.clip.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self.height = int(self.clip.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

        # Read in all frames
        self.frames = []
        while self.clip.isOpened():
            ret, frame = self.clip.read()
            if not ret:
                break
            self.frames.append(frame)
        self.clip.release()

        super(VideoStim, self).__init__(filename)

    def _extract_duration(self):
        self.duration = self.n_frames * 1. / self.fps

    def __iter__(self):
        """ Frame iteration. """
        for i, f in enumerate(self.frames):
            yield VideoFrameStim(self, i, data=f)

    def annotate(self, annotators, merge_events=True, **kwargs):
        period = 1. / self.fps
        timeline = Timeline(period=period)
        for ann in annotators:
            # For VideoAnnotators, pass the entire stim
            if ann.target.__name__ == self.__class__.__name__:
                events = ann.apply(self, **kwargs)
                for ev in events:
                    timeline.add_event(ev, merge=merge_events)
            # Otherwise, for images, loop over frames
            else:
                c = 0
                for frame in self:
                    if frame.data is not None:
                        event = Event(onset=c * period)
                        event.add_note(ann.apply(frame))
                        timeline.add_event(event, merge=merge_events)
                        c += 1
        return timeline


class AudioStim(DynamicStim):

    ''' An audio clip. For now, only handles wav files. '''

    def __init__(self, filename):
        self.sampling_rate, self.data = wavfile.read(filename)
        self._extract_duration()
        super(AudioStim, self).__init__(filename)

    def _extract_duration(self):
        self.duration = len(self.data)*1./self.sampling_rate

    def annotate(self, annotators, merge_events=True):
        timeline = Timeline()
        for ann in annotators:
            events = ann.apply(self)
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

    def annotate(self, annotators):
        timeline = Timeline()
        audio_anns, text_anns = [], []
        for ann in annotators:
            if ann.target.__name__ == 'AudioStim':
                audio_anns.append(ann)
            elif ann.target.__name__ == 'ComplexTextStim':
                text_anns.append(ann)

        audio_tl = super(TranscribedAudioStim, self).annotate(audio_anns)
        timeline.merge(audio_tl)
        text_tl = self.transcription.annotate(text_anns)
        timeline.merge(text_tl)
        return timeline


class TextStim(Stim):

    ''' Any text stimulus. '''
    def __init__(self, text):
        self.text = text


class DynamicTextStim(TextStim):

    ''' A text stimulus with timing/onset information. '''

    def __init__(self, text, order, onset=None, duration=None):
        self.order = order
        self.onset = onset
        self.duration = duration
        super(DynamicTextStim, self).__init__(text)


class ComplexTextStim(object):

    ''' A collection of text stims (e.g., a story), typically ordered and with
    onsets and/or durations associated with each element.
    Args:
        filename (str): The filename to read from. Must be tab-delimited text.
            Files must always contain a column containing the text of each
            stimulus in the collection. Optionally, additional columns can be
            included that contain duration and onset information. If a header
            row is present in the file, valid columns must be labeled as
            'text', 'onset', and 'duration' where available (though only text
            is mandatory). If no header is present in the file, the columns
            argument will be used to infer the indices of the key columns.
        columns (str): Optional specification of column order. An abbreviated
            string denoting the column position of text, onset, and duration
            in the file. Use t for text, o for onset, d for duration. For
            example, passing 'ot' indicates that the first column contains
            the onsets and the second contains the text. Passing 'tod'
            indicates that the first three columns contain text, onset, and
            duration information, respectively. Note that if the input file
            contains a header row, the columns argument will be ignored.
        default_duration (float): the duration to assign to any text elements
            in the collection that do not have an explicit value provided
            in the input file.
    '''

    def __init__(self, filename, columns='tod', default_duration=None):

        self.elements = []
        tod_names = {'t': 'text', 'o': 'onset', 'd': 'duration'}

        first_row = open(filename).readline().strip().split('\t')
        if len(set(first_row) & set(tod_names.values())):
            col_names = None
        else:
            col_names = [tod_names[x] for x in columns]

        data = pd.read_csv(filename, sep='\t', names=col_names)

        for i, r in data.iterrows():
            if 'onset' not in r:
                elem = TextStim(r['text'])
            else:
                duration = r.get('duration', None)
                if duration is None:
                    duration = default_duration
                elem = DynamicTextStim(r['text'], i, r['onset'], duration)
            self.elements.append(elem)

    def __iter__(self):
        """ Iterate text elements. """
        for elem in self.elements:
            yield elem

    def annotate(self, annotators, merge_events=True):
        timeline = Timeline()
        for ann in annotators:
            if ann.target.__name__ == self.__class__.__name__:
                events = ann.apply(self)
                for ev in events:
                    timeline.add_event(ev, merge=merge_events)
            else:
                for elem in self.elements:
                    event = Event(onset=elem.onset)
                    event.add_note(ann.apply(elem))
                    timeline.add_event(event, merge=merge_events)
        return timeline

    @classmethod
    def from_text(cls, text, unit='word'):
        """ Initialize from a single string, by automatically segmenting into
        individual strings.
        """
        pass

