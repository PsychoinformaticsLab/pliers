from abc import ABCMeta, abstractmethod, abstractproperty
import cv2
import numpy as np
from .core import Note, Event, Timeline
import stims


class DynamicAnnotatorMixin(object):
    pass


class StaticAnnotatorMixin(object):
    pass


class Annotator(object):

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self):
        pass

    # @abstractproperty
    # def stim_types(self):
    #     pass


class ImageAnnotator(Annotator, StaticAnnotatorMixin):

    stim_types = [stims.ImageStim]


class VideoAnnotator(Annotator, DynamicAnnotatorMixin):

    stim_types = [stims.VideoStim]


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


class TextAnnotator(DynamicAnnotatorMixin, Annotator):
    pass


class CornerDetectionAnnotator(ImageAnnotator):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.fast = cv2.FastFeatureDetector()

    def apply(self, img):
        kp = self.fast.detect(img, None)
        return Note(img, self, {'corners_detected': kp})


# class ImageFeatureDetectionAnnotator(ImageAnnotator):

#     def __init__(self):
#         super(self.__class__, self).__init__()
#         self.star = cv2.FeatureDetector_create("STAR")
#         self.brief = cv2.DescriptorExtractor_create("BRIEF")
#         self.name = "FeatureDetection"

#     def apply(self, img):
#         kp = self.star.detect(img, None)
#         kp, des = self.brief.compute(img, kp)
#         return Note(img, self, {'features_detected': des})


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


# class ShapeMotionDetectionAnnotator(VideoAnnotator):

#     def __init__(self):
#         super(ShapeMotionDetectionAnnotator, self).__init__()
#         self.name = "ShapeMotionDetection"

#     def apply(self, video):

#         def rgb2gray(rgb):
#             r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
#             gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#             return gray

#         img = video.frames[50]
#         for img in video.frames:
#             if img is not None:
#                 gray = 255 - rgb2gray(img).astype('uint8')
#                 ret, thresh = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
#                 thresh = np.copy(thresh)
#                 contours, h = contours, hierarchy = cv2.findContours(
#                     thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#                 cv2.drawContours(img, contours, -1, (200, 255, 200), 2)
#                 cv2.imshow('frame', img)
#                 cv2.waitKey(20)
