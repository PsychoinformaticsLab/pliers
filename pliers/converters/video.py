''' Converter classes that operate on VideoStim inputs. '''

import os
from pliers.stimuli.video import VideoStim, DerivedVideoStim, VideoFrameStim
from pliers.stimuli.audio import AudioStim
from pliers.utils import progress_bar_wrapper
from .base import Converter


class VideoToAudioConverter(Converter):

    ''' Convert a VideoStim to an AudioStim by extracting the audio track
    using moviepy. '''
    _input_type = VideoStim
    _output_type = AudioStim

    def _convert(self, video):
        return AudioStim(clip=video.clip.audio)


class VideoToDerivedVideoConverter(Converter):

    ''' Base VideoToDerivedVideo Converter class; all subclasses can only be
    applied to video and convert to derived (sampled) video. '''
    _input_type = VideoStim
    _output_type = DerivedVideoStim


class FrameSamplingConverter(VideoToDerivedVideoConverter):

    ''' Samples frames from video stimuli, to improve efficiency.

    Args:
        every (int): takes every nth frame
        hertz (int): takes n frames per second
        top_n (int): takes top n frames sorted by the absolute difference
         with the next frame
    '''

    _log_attributes = ('every', 'hertz', 'top_n')

    def __init__(self, every=None, hertz=None, top_n=None):
        if every is None and hertz is None and top_n is None:
            raise ValueError("When initializing the FrameSamplingConverter, "
                             "one of the 'every', 'hertz', or 'top_n' must "
                             "be specified.")
        super(FrameSamplingConverter, self).__init__()
        self.every = every
        self.hertz = hertz
        self.top_n = top_n

    def _convert(self, video):
        if not hasattr(video, "frame_index"):
            frame_index = range(video.n_frames)
        else:
            frame_index = video.frame_index

        if self.every is not None:
            new_idx = range(video.n_frames)[::self.every]
        elif self.hertz is not None:
            interval = int(video.fps / self.hertz)
            new_idx = range(video.n_frames)[::interval]
        elif self.top_n is not None:
            import cv2
            diffs = []
            for i, img in enumerate(video.frames):
                if i == 0:
                    last = img
                    continue
                diffs.append(sum(cv2.sumElems(cv2.absdiff(last, img))))
                last = img
            new_idx = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[
                :self.top_n]

        frame_index = sorted(list(set(frame_index).intersection(new_idx)))

        # Construct new VideoFrameStim for each frame index
        onsets = [frame_num * (1. / video.fps) for frame_num in frame_index]
        frames = []
        for i, f in progress_bar_wrapper(enumerate(frame_index),
                                         desc='Video frame',
                                         total=len(frame_index)):
            if f != frame_index[-1]:
                dur = onsets[i+1] - onsets[i]
            else:
                dur = (video.n_frames / video.fps) - onsets[i]

            elem = VideoFrameStim(video=video, frame_num=f, duration=dur)
            frames.append(elem)

        return DerivedVideoStim(filename=video.filename, frames=frames,
                                frame_index=frame_index)
