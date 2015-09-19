from featurex.stimuli.video import VideoStim
from featurex.extractors import StimExtractor
from featurex.core import Value, Event
import cv2
import numpy as np


class VideoExtractor(StimExtractor):

    target = VideoStim


class DenseOpticalFlowExtractor(VideoExtractor):

    def __init__(self):
        super(self.__class__, self).__init__()

    def apply(self, video, show=False):

        events = []
        for i, f in enumerate(video):

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

            value = Value(video, self, {'total_flow': total_flow})
            event = Event(onset=f.onset, duration=f.duration, values=[value])
            events.append(event)

        return events
