from featurex.stimuli.video import VideoStim, DerivedVideoStim, VideoFrameStim
from featurex.converters import Converter

import pandas as pd


class VideoToDerivedVideoConverter(Converter):

    ''' Base AudioToText Converter class; all subclasses can only be applied to
    audio and convert to text. '''
    target = VideoStim
    _output_type = DerivedVideoStim


class FrameSamplingConverter(VideoToDerivedVideoConverter):
    ''' 
    Samples frames from video stimuli, to improve efficiency

    Args:
        every (int): takes every nth frame
        hertz (int): takes n frames per second
        num_frames (int): takes top n frames sorted by the absolute difference
         with the next frame
    '''
    def __init__(self, every=None, hertz=None, num_frames=None):
        self.every = every
        self.hertz = hertz
        self.num_frames = num_frames

    def _convert(self, video):
        if not hasattr(video, "frame_index"):
            frame_index = range(video.n_frames)
        else:
            frame_index = video.frame_index
        
        if not hasattr(video, "history"):
            history = pd.DataFrame(columns=["filter", "value", "n_frames"])
        else:
            history = video.history

        if self.every is not None:
            new_idx = range(video.n_frames)[::self.every]
            history.loc[history.shape[0]]= ["every", self.every, len(new_idx)]
        elif self.hertz is not None:
            interval = int(video.fps / self.hertz)
            new_idx = range(video.n_frames)[::interval]
            history.loc[history.shape[0]] = ["hertz", self.hertz, len(new_idx)]
        elif self.num_frames is not None:
            import cv2
            diffs = []
            for i, img in enumerate(video.frames):
                if i == 0:
                    last = img
                    continue
                diffs.append(sum(cv2.sumElems(cv2.absdiff(last, img))))
                last = img
            new_idx = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[:self.num_frames]
            history.loc[history.shape[0]] = ["num_frames", self.num_frames, len(new_idx)]

        frame_index = sorted(list(set(frame_index).intersection(new_idx)))

        # Construct new VideoFrameStim for each frame index
        onsets = [frame_num * (1. / video.fps) for frame_num in frame_index]
        elements = []
        for i, f in enumerate(frame_index):
            if f != frame_index[-1]:
                dur = onsets[i+1] - onsets[i]
            else:
                dur = (len(video.frames) / video.fps) - onsets[i]

            elem = VideoFrameStim(video=video.clip, frame_num=f,
                                  duration=dur)
            elements.append(elem)

        return DerivedVideoStim(filename=video.filename,
                                elements=elements,
                                frame_index=frame_index,
                                history=history)
        