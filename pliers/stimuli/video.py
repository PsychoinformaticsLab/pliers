''' Classes that represent video clips. '''

from __future__ import division
from math import ceil
from moviepy.video.io.VideoFileClip import VideoFileClip
from .base import Stim
from .audio import AudioStim
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
        if video.onset:
            onset += video.onset
        super(VideoFrameStim, self).__init__(filename, onset, duration, data)
        if data is None:
            self.data = self.video.get_frame(index=frame_num).data
        self.name += 'frame[%s]' % frame_num


class VideoFrameCollectionStim(Stim):

    ''' A collection of video frames.
    Args:
        filename (str): Path to input file, if one exists.
        frame_index (list): List of indices of frames retained from the
            original video. Uses every frame by default
            (i.e. for normal VideoStims).
        onset (float): Optional onset of the video file (in seconds) with
            respect to some more general context or timeline the user wishes
            to keep track of.
        url (str): Optional url source for a video.
    '''

    _default_file_extension = '.mp4'

    def __init__(self, filename=None, frame_index=None, onset=None, url=None):
        if url is not None:
            filename = url
        self.filename = filename
        self._load_clip()
        self.fps = self.clip.fps
        self.width = self.clip.w
        self.height = self.clip.h
        if frame_index:
            self.frame_index = frame_index
        else:
            self.frame_index = range(int(ceil(self.fps * self.clip.duration)))
        self.n_frames = len(self.frame_index)
        duration = self.clip.duration
        super(VideoFrameCollectionStim, self).__init__(filename,
                                                       onset=onset,
                                                       duration=duration)

    def _load_clip(self):
        audio_fps = AudioStim.get_sampling_rate(self.filename)
        self.clip = VideoFileClip(self.filename, audio_fps=audio_fps)

    def __iter__(self):
        """ Frame iteration. """
        for i, f in enumerate(self.frame_index):
            yield self.get_frame(i)

    @property
    def frames(self):
        return (f for f in self)

    def get_frame(self, index=None, onset=None):
        ''' Get video frame at the specified index.
        Args:
            index (int): Positional index of the desired frame.
            onset (float): Onset (in seconds) of the desired frame.
        '''
        if onset:
            index = int(onset * self.fps)

        frame_num = self.frame_index[index]
        onset = float(frame_num) / self.fps

        if index < self.n_frames - 1:
            next_frame_num = self.frame_index[index + 1]
            end = float(next_frame_num) / self.fps
        else:
            end = float(self.duration)

        duration = end - onset if end > onset else 0.0

        return VideoFrameStim(self, frame_num,
                              data=self.clip.get_frame(onset),
                              duration=duration)

    def __getstate__(self):
        d = self.__dict__.copy()
        d['clip'] = None
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._load_clip()

    def save(self, path):
        ''' Save source video to file.
        Args:
            path (str): Filename to save to.
        Notes: Saves entire source video to file, not just currently selected
            frames.
        '''
        # IMPORTANT WARNING: saves entire source video
        self.clip.write_videofile(path)


class VideoStim(VideoFrameCollectionStim):

    ''' A video.
    Args:
        filename (str): Path to input file, if one exists.
        onset (float): Optional onset of the video file (in seconds) with
            respect to some more general context or timeline the user wishes
            to keep track of.
        url (str): Optional url source for a video.
    '''

    def __init__(self, filename=None, onset=None, url=None):
        super(VideoStim, self).__init__(filename=filename,
                                        onset=onset,
                                        url=url)
