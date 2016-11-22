from featurex.stimuli.video import VideoStim, DerivedVideoStim, VideoFrameStim
from featurex.stimuli.audio import AudioStim
from featurex.converters import Converter

import pandas as pd
import os


class VideoToAudioConverter(Converter):

    ''' Base VideoToDerivedVideo Converter class; all subclasses can only be
    applied to video and convert to derived (sampled) video. '''
    target = VideoStim
    _output_type = AudioStim

    def __init__(self, fps=44100, nbytes=2,
                    buffersize=2000, bitrate=None,
                    ffmpeg_params=None):
        super(self.__class__, self).__init__()
        self.fps = fps
        self.nbytes = nbytes
        self.buffersize = buffersize
        self.bitrate = bitrate
        self.ffmpeg_params = ffmpeg_params

    def _convert(self, video):
        filename = os.path.splitext(video.filename)[0] + '.wav'
        video.clip.audio.write_audiofile(filename, self.fps, self.nbytes,
                                        self.buffersize, self.bitrate,
                                        self.ffmpeg_params)
        return AudioStim(filename)


class VideoToDerivedVideoConverter(Converter):

    ''' Base VideoToDerivedVideo Converter class; all subclasses can only be
    applied to video and convert to derived (sampled) video. '''
    target = VideoStim
    _output_type = DerivedVideoStim


class FrameSamplingConverter(VideoToDerivedVideoConverter):
    ''' 
    Samples frames from video stimuli, to improve efficiency

    Args:
        every (int): takes every nth frame
        hertz (int): takes n frames per second
        top_n (int): takes top n frames sorted by the absolute difference
         with the next frame
    '''
    def __init__(self, every=None, hertz=None, top_n=None):
        self.every = every
        self.hertz = hertz
        self.top_n = top_n

    def _convert(self, video):
        if not hasattr(video, "frame_index"):
            frame_index = range(video.n_frames)
        else:
            frame_index = video.frame_index
        
        if not hasattr(video, "history"):
            history = pd.DataFrame(columns=["filter", "value", "n_frames"])
        else:
            history = video.history.copy()

        if self.every is not None:
            new_idx = range(video.n_frames)[::self.every]
            history.loc[history.shape[0]]= ["every", self.every, len(new_idx)]
        elif self.hertz is not None:
            interval = int(video.fps / self.hertz)
            new_idx = range(video.n_frames)[::interval]
            history.loc[history.shape[0]] = ["hertz", self.hertz, len(new_idx)]
        elif self.top_n is not None:
            import cv2
            diffs = []
            for i, img in enumerate(video.frames):
                if i == 0:
                    last = img
                    continue
                diffs.append(sum(cv2.sumElems(cv2.absdiff(last, img))))
                last = img
            new_idx = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[:self.top_n]
            history.loc[history.shape[0]] = ["top_n", self.top_n, len(new_idx)]

        frame_index = sorted(list(set(frame_index).intersection(new_idx)))

        # Construct new VideoFrameStim for each frame index
        onsets = [frame_num * (1. / video.fps) for frame_num in frame_index]
        elements = []
        for i, f in enumerate(frame_index):
            if f != frame_index[-1]:
                dur = onsets[i+1] - onsets[i]
            else:
                dur = (len(video.frames) / video.fps) - onsets[i]

            elem = VideoFrameStim(video=video, frame_num=f,
                                  duration=dur)
            elements.append(elem)

        return DerivedVideoStim(filename=video.filename,
                                elements=elements,
                                frame_index=frame_index,
                                history=history)
        