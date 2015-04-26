from abc import ABCMeta, abstractmethod, abstractproperty
import cv2
import numpy as np
import pandas as pd
from .core import Note
import stims


class Annotator(object):

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self):
        pass

    @abstractproperty
    def target(self):
        pass


class ImageAnnotator(Annotator):

    target = stims.ImageStim


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


class TextAnnotator(Annotator):
 
    target = stims.TextStim


class TextDictionaryAnnotator(TextAnnotator):

    def __init__(self, dictionary, variables=None, missing='nan'):
        self.data = pd.read_csv(dictionary, sep='\t', index_col=0)
        self.variables = variables
        if variables is not None:
            self.data = self.data[variables]
        # Set up response when key is missing
        self.missing = np.nan
        super(self.__class__, self).__init__()

    def apply(self, stim):
        if stim.text not in self.data.index:
            vals = pd.Series(self.missing, self.variables)
        else:
            vals = self.data.loc[stim.text]
        return Note(stim, self, vals.to_dict())


class CornerDetectionAnnotator(ImageAnnotator):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.fast = cv2.FastFeatureDetector()

    def apply(self, img):
        kp = self.fast.detect(img, None)
        return Note(img, self, {'corners_detected': kp})


class FaceDetectionAnnotator(ImageAnnotator):

    def __init__(self):
        self.cascade = cv2.CascadeClassifier('/Users/tal/Downloads/cascade.xml')
        super(self.__class__, self).__init__()

    def apply(self, img, show=False):
        data = img.data
        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        if show:
            for (x, y, w, h) in faces:
                cv2.rectangle(data, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('frame', data)
            cv2.waitKey(1)

        return Note(img, self, {'num_faces': len(faces)})

