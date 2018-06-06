''' Filters that operate on TextStim inputs. '''

from pliers.stimuli.video import VideoStim, VideoFrameCollectionStim
from pliers.utils import attempt_to_import, verify_dependencies
from .base import Filter, TemporalTrimmingFilter

import numpy as np


cv2 = attempt_to_import('cv2')


class VideoFilter(Filter):

    ''' Base class for all VideoFilters. '''

    _input_type = VideoStim


class FrameSamplingFilter(Filter):

    ''' Samples frames from video stimuli, to improve efficiency.

    Args:
        every (int): takes every nth frame
        hertz (int): takes n frames per second
        top_n (int): takes top n frames sorted by the absolute difference
         with the next frame
    '''

    _input_type = VideoFrameCollectionStim
    _log_attributes = ('every', 'hertz', 'top_n')
    VERSION = '1.0'

    def __init__(self, every=None, hertz=None, top_n=None):
        if every is None and hertz is None and top_n is None:
            raise ValueError("When initializing the FrameSamplingFilter, "
                             "one of the 'every', 'hertz', or 'top_n' must "
                             "be specified.")
        self.every = every
        self.hertz = hertz
        self.top_n = top_n
        super(FrameSamplingFilter, self).__init__()

    def _filter(self, video):
        if not isinstance(video, VideoStim):
            raise TypeError('Currently, frame sampling is only supported for '
                            'complete VideoStim inputs.')

        if self.every is not None:
            new_idx = range(video.n_frames)[::self.every]
        elif self.hertz is not None:
            interval = video.fps / float(self.hertz)
            new_idx = np.arange(0, video.n_frames, interval).astype(int)
            new_idx = list(new_idx)
        elif self.top_n is not None:
            verify_dependencies(['cv2'])
            diffs = []
            for i, img in enumerate(video.frames):
                if i == 0:
                    last = img
                    continue
                pixel_diffs = cv2.sumElems(cv2.absdiff(last.data, img.data))
                diffs.append(sum(pixel_diffs))
                last = img
            new_idx = sorted(range(len(diffs)),
                             key=lambda i: diffs[i],
                             reverse=True)[:self.top_n]

        return VideoFrameCollectionStim(filename=video.filename,
                                        clip=video.clip,
                                        frame_index=new_idx)


class VideoTrimmingFilter(TemporalTrimmingFilter, VideoFilter):
    pass
