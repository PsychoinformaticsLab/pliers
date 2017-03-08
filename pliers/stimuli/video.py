''' Classes that represent video clips. '''

from __future__ import division
from moviepy.video.io.VideoFileClip import VideoFileClip
from .base import Stim, CollectionStimMixin
from .image import ImageStim


class VideoFrameStim(ImageStim):

    ''' A single frame of video.
    Args:
        video (VideoStim): The source VideoStim the frame is drawn from.
        frame_num (int): The index of the current frame in the source video.
        duration (float): Optional duration of presentation, in seconds.
        filename (str): Path to input video file, if one exists.
        data (ndarray): Optional numpy array to initialize the image from.
    '''

    def __init__(self, video, frame_num, duration=None, filename=None,
                 data=None):
        self.video = video
        self.frame_num = frame_num
        spf = 1. / video.fps
        duration = spf if duration is None else duration
        onset = frame_num * spf
        super(VideoFrameStim, self).__init__(filename, onset, duration, data)
        if data is None:
            self.data = self.video.get_frame(index=frame_num).data
        self.name += 'frame[%s]' % frame_num


class VideoStim(Stim, CollectionStimMixin):

    ''' A video.
    Args:
        filename (str): Path to input file, if one exists.
        onset (float): Optional onset of the video file (in seconds) with
            respect to some more general context or timeline the user wishes
            to keep track of.
    '''

    def __init__(self, filename=None, onset=None, url=None):
        if url is not None:
            filename = url
        self.filename = filename
        self._load_clip()
        self.fps = self.clip.fps
        self.width = self.clip.w
        self.height = self.clip.h
        self.n_frames = int(self.fps * self.clip.duration)
        duration = self.clip.duration

        super(VideoStim, self).__init__(filename, onset, duration)

    def _load_clip(self):
        self.clip = VideoFileClip(self.filename)

    def __iter__(self):
        """ Frame iteration. """
        for i, f in enumerate(self.clip.iter_frames()):
            yield VideoFrameStim(self, i, data=f)

    def __getstate__(self):
        d = self.__dict__.copy()
        d['clip'] = None
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._load_clip()

    @property
    def frames(self):
        return (f for f in self.clip.iter_frames())

    def get_frame(self, index=None, onset=None):
        if index is not None:
            onset = float(index) / self.fps
        else:
            index = int(onset * self.fps)
        return VideoFrameStim(self, index, data=self.clip.get_frame(onset))


class DerivedVideoStim(VideoStim):

    """
    VideoStim containing keyframes (for API calls). Each keyframe is associated
    with a duration reflecting the length of its "scene."
    Args:
        filename (str): Path to input file, if one exists.
        frames (iterable): iterable of frames retained from original VideoStim.
        frame_index (list): List of indices of frames retained from the
            original VideoStim.
    """

    def __init__(self, filename, frames, frame_index=None):
        super(DerivedVideoStim, self).__init__(filename)
        self._frames = frames
        self.frame_index = frame_index
        self.name += '_derived'

    @property
    def frames(self):
        return (f for f in self._frames)

    def __iter__(self):
        return self.frames
