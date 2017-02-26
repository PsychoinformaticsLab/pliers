'''
Extractors that operate primarily or exclusively on Video stimuli.
'''

from pliers.stimuli.video import VideoStim
from pliers.extractors.base import Extractor, ExtractorResult

import numpy as np

# Optional dependencies
try:
    import cv2
except ImportError:
    pass


class VideoExtractor(Extractor):

    ''' Base Video Extractor class; all subclasses can only be applied to
    video. '''
    _input_type = VideoStim


class DenseOpticalFlowExtractor(VideoExtractor):

    ''' Extracts total amount of optical flow between every pair of video
    frames.
    '''

    def __init__(self, show=False):
        super(self.__class__, self).__init__()
        self.show = show

    def _extract(self, stim):

        flows = []
        onsets = []
        durations = []
        for i, f in enumerate(stim):

            img = f.data
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if i == 0:
                last_frame = img

            flow = cv2.calcOpticalFlowFarneback(
                last_frame, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = np.sqrt((flow ** 2).sum(2))

            if self.show:
                cv2.imshow('frame', flow.astype('int8'))
                cv2.waitKey(1)

            last_frame = img
            flows.append(flow.sum())
            onsets.append(f.onset)
            durations.append(f.duration)

        return ExtractorResult(flows, stim, self, features=['total_flow'],
                               onsets=onsets, durations=durations)
