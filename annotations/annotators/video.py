from annotations import stims
from annotations.annotators import Annotator
from annotations.annotators.image import ImageAnnotator
from annotations.core import Note
import cv2
import numpy as np


class VideoAnnotator(Annotator):

    target = stims.VideoStim


class DenseOpticalFlowAnnotator(ImageAnnotator):

    def __init__(self):
        self.last_frame = None
        super(self.__class__, self).__init__()

    def apply(self, img, show=False):
        _img = cv2.cvtColor(img.data, cv2.COLOR_BGR2GRAY)
        if self.last_frame is None:
            self.last_frame = _img
            total_flow = 0
        else:
            curr_frame = _img
            flow = cv2.calcOpticalFlowFarneback(
                self.last_frame, curr_frame, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = np.sqrt((flow**2).sum(2))

            if show:
                cv2.imshow('frame', flow.astype('int8'))
                cv2.waitKey(1)

            self.last_frame = curr_frame
            total_flow = flow.sum()

        return Note(img, self, {'total_flow': total_flow})
