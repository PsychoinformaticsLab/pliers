'''
Extractors that operate primarily or exclusively on Video stimuli.
'''

from featurex.stimuli.video import VideoStim
from featurex.extractors import StimExtractor
from featurex.core import Value, Event
import numpy as np

# Optional dependencies
try:
    import cv2
except ImportError:
    pass


class VideoExtractor(StimExtractor):
    ''' Base Video Extractor class; all subclasses can only be applied to
    video. '''
    target = VideoStim


class DenseOpticalFlowExtractor(VideoExtractor):
    ''' Extracts total amount of optical flow between every pair of video
    frames.

    '''

    def apply(self, stim, show=False):

        events = []
        for i, f in enumerate(stim):

            img = f.data
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if i == 0:
                last_frame = img
                total_flow = 0

            flow = cv2.calcOpticalFlowFarneback(
                last_frame, img, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = np.sqrt((flow ** 2).sum(2))

            if show:
                cv2.imshow('frame', flow.astype('int8'))
                cv2.waitKey(1)

            last_frame = img
            total_flow = flow.sum()

            value = Value(stim, self, {'total_flow': total_flow})
            event = Event(onset=f.onset, duration=f.duration, values=[value])
            events.append(event)

        return events
