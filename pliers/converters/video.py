''' Converter classes that operate on VideoStim inputs. '''

from pliers.stimuli.video import VideoStim, VideoFrameCollectionStim
from pliers.stimuli.audio import AudioStim
from .base import Converter


class VideoToAudioConverter(Converter):

    ''' Convert a VideoStim to an AudioStim by extracting the audio track
    using moviepy. '''
    _input_type = VideoStim
    _output_type = AudioStim

    def _convert(self, video):
        return AudioStim(clip=video.clip.audio, onset=video.onset)


class VideoToDerivedVideoConverter(Converter):

    ''' Base VideoToDerivedVideo Converter class; all subclasses can only be
    applied to video and convert to derived (sampled) video. '''
    _input_type = VideoFrameCollectionStim
    _output_type = VideoFrameCollectionStim


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
        if self.every is not None:
            new_idx = range(int(video.fps * video.clip.duration))[::self.every]
        elif self.hertz is not None:
            interval = int(video.fps / self.hertz)
            new_idx = range(int(video.fps * video.clip.duration))[::interval]
            print video.frame_index
            print new_idx
        elif self.top_n is not None:
            import cv2
            diffs = []
            for i, img in enumerate(video.frames):
                if i == 0:
                    last = img
                    continue
                diffs.append(sum(cv2.sumElems(cv2.absdiff(last.data, img.data))))
                last = img
            new_idx = sorted(range(len(diffs)),
                             key=lambda i: diffs[i],
                             reverse=True)[:self.top_n]

        frame_index = sorted(list(set(video.frame_index).intersection(new_idx)))

        return VideoFrameCollectionStim(filename=video.filename,
                                        frame_index=frame_index,
                                        onset=video.onset)
