from featurex.stimuli import Stim, DynamicStim
from featurex.core import Timeline, Event
import six

# Optional dependencies
try:
    import cv2
except ImportError:
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
        self.fps = self.clip.get(cv2.CAP_PROP_FPS)
        self.n_frames = self.clip.get(cv2.CAP_PROP_FRAME_COUNT)
        self.width = int(self.clip.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.clip.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    def extract(self, extractors, merge_events=True, **kwargs):
        period = 1. / self.fps
        timeline = Timeline(period=period)
        for ext in extractors:
            # For VideoExtractors, pass the entire stim
            if ext.target.__name__ == self.__class__.__name__:
                events = ext.apply(self, **kwargs)
                for ev in events:
                    timeline.add_event(ev, merge=merge_events)
            # Otherwise, for images, loop over frames
            else:
                c = 0
                for frame in self:
                    if frame.data is not None:
                        event = Event(onset=c * period)
                        event.add_value(ext.apply(frame))
                        timeline.add_event(event, merge=merge_events)
                        c += 1
        return timeline
