from featurex.stimuli import video
from featurex.extractors import StimExtractor
import cv2
import time
import numpy as np
from featurex.core import Value

from metamind.api import set_api_key, general_image_classifier, ClassificationModel


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


class BrightnessExtractor(ImageExtractor):
    ''' Gets the average luminosity of the pixels in the image '''

    def __init__(self):
        super(self.__class__, self).__init__()

    def apply(self, img):
        data = img.data
        hsv = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
        avg_brightness = hsv[:,:,2].mean()

        return Value(img, self, {'avg_brightness': avg_brightness})

class SharpnessExtractor(ImageExtractor):
    ''' Gets the degree of blur/sharpness of the image '''
    def __init__(self):
        super(self.__class__, self).__init__()

    def apply(self, img):
        # Taken from http://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry?lq=1
        # I don't understand the math behind this
        data = img.data
        gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY) 
        
        sharpness = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray_image, 3)))
        return Value(img, self, {'sharpness': sharpness})

class MetamindFeaturesExtractor(ImageExtractor):
    ''' Uses the MetaMind API to extract features with an existing classifier '''
    def __init__(self):
        ImageExtractor.__init__(self)
        set_api_key('1s8nqbHlFfPf82IrDlGFmz2uEXHlSJ6DveJx7r8Ycoz8ahqBwq')
        self.classifier = general_image_classifier

    def apply(self, img):
        data = img.data
        temp_file = 'temp.jpg'
        cv2.imwrite(temp_file, data)
        labels = self.classifier.predict(temp_file, input_type='files')
        top_label = labels[0]['label']
        print top_label
        time.sleep(1.0)

        return Value(img, self, {'labels': labels})

class IndoorOutdoorExtractor(MetamindFeaturesExtractor):
    ''' Classify if the image is indoor or outdoor '''
    def __init__(self):
        super(self.__class__, self).__init__()
        self.classifier = ClassificationModel(id=25463)

class FacialExpressionExtractor(MetamindFeaturesExtractor):
    ''' Classify if the image for facial expressions '''
    def __init__(self):
        super(self.__class__, self).__init__()
        self.classifier = ClassificationModel(id=30198)
