from abc import ABCMeta, abstractmethod
import cv2
import six
from .core import Timeline, Event, Note
import pandas as pd


class Stim(object):

    __metaclass__ = ABCMeta

    def __init__(self, filename, label, description):

        self.filename = filename
        self.label = label
        self.description = description
        self.annotations = []


class DynamicStim(Stim):

    ''' Any Stim that has as a temporal dimension. '''

    __metaclass__ = ABCMeta

    def __init__(self, filename, label, description):
        super(DynamicStim, self).__init__(filename, label, description)
        self._extract_duration()

    @abstractmethod
    def _extract_duration(self):
        pass


class ImageStim(Stim):

    ''' A static image. '''

    def __init__(self, filename=None, data=None, label=None, duration=None,
                 description=None):
        if data is None and isinstance(filename, six.string_types):
            data = cv2.imread(filename)
        super(ImageStim, self).__init__(filename, label, description)
        self.data = data
        self.duration = duration


class VideoFrameStim(ImageStim):

    ''' A single frame of video. '''

    def __init__(self, video, frame_num, filename=None, data=None, label=None,
                 description=None):
        super(VideoFrameStim, self).__init__(filename, data, label,
                                             description)
        self.video = video
        self.frame_num = frame_num
        self.duration = 1. / video.fps
        self.onset = frame_num * self.duration


class VideoStim(DynamicStim):

    ''' A video. '''

    def __init__(self, filename, label=None, description=None):
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

        super(VideoStim, self).__init__(filename, label, description)

    def _extract_duration(self):
        self.duration = self.n_frames * 1. / self.fps

    def __iter__(self):
        """ Frame iteration. """
        for i, f in enumerate(self.frames):
            yield VideoFrameStim(self, i, data=f)

    def annotate(self, annotators, merge_events=True):
        period = 1. / self.fps
        timeline = Timeline(period=period)
        for ann in annotators:
            if ann.target.__name__ == self.__class__.__name__:
                events = ann.apply(self)
                for ev in events:
                    timeline.add_event(ev, merge=merge_events)
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

    ''' An audio clip. '''

    def __init__(self, filename, label=None, description=None):
        super(VideoStim, self).__init__(filename, label, description)

    def _extract_duration(self):
        pass


class TextStim(object):

    ''' Any text stimulus. '''
    def __init__(self, text):
        self.text = text


class DynamicTextStim(TextStim):

    ''' A text stimulus with timing/onset information. '''

    def __init__(self, text, order, onset=None, duration=None):
        super(DynamicTextStim, self).__init__(text)
        self.order = order
        self.onset = onset
        self.duration = duration


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
            in the input filename.
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


class StimCollection(object):
    pass
