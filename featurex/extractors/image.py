from featurex.stimuli import video
from featurex.extractors import StimExtractor
import cv2
from featurex.core import Value


class ImageExtractor(StimExtractor):
    ''' Base Image Extractor class; all subclasses can only be applied to
    images. '''
    target = video.ImageStim


class CornerDetectionExtractor(ImageExtractor):
    ''' Wraps OpenCV's FastFeatureDetector; should not be used for anything
    important yet. '''

    def __init__(self):
        super(self.__class__, self).__init__()
        self.fast = cv2.FastFeatureDetector()

    def apply(self, img):
        kp = self.fast.detect(img, None)
        return Value(img, self, {'corners_detected': kp})


class FaceDetectionExtractor(ImageExtractor):
    ''' Face detection based on OpenCV's CascadeClassifier. This will generally
    not work well without training, and should not be used for anything
    important at the moment. '''
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            '/Users/tal/Downloads/cascade.xml')
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
                cv2.rectangle(data, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('frame', data)
            cv2.waitKey(1)

        return Value(img, self, {'num_faces': len(faces)})
