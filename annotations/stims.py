from abc import ABCMeta, abstractmethod
import cv2
import six


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
        self.onset = frame_num * 1. / video.fps


class VideoStim(DynamicStim):
    ''' A video. '''
    def __init__(self, filename, label=None, description=None):
        self.clip = cv2.VideoCapture(filename)
        self.fps = self.clip.get(cv2.cv.CV_CAP_PROP_FPS)
        self.n_frames = self.clip.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        self.width = int(self.clip.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self.height = int(self.clip.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        super(VideoStim, self).__init__(filename, label, description)

    def _extract_duration(self):
        self.duration = self.n_frames * 1. / self.fps

    def __iter__(self):
        """ Frame iteration. """
        i = 0
        while self.clip.isOpened():
            ret, frame = self.clip.read()
            yield VideoFrameStim(self, i, data=frame)
            if not ret:
                break
            i += 1
        self.clip.release()


class AudioStim(DynamicStim):
    ''' An audio clip. '''
    def __init__(self, filename, label=None, description=None):
        super(VideoStim, self).__init__(filename, label, description)

    def _extract_duration(self):
        pass


class TextStim(Stim):
    ''' Any text stimulus. '''
    pass


class DynamicTextStim(DynamicStim, TextStim):
    ''' A text stimulus with timing information. '''
    pass


class StimCollection(object):
    pass
