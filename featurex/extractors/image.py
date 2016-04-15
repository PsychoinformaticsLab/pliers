from featurex.stimuli import video
from featurex.extractors import StimExtractor
from featurex.core import Value
import time
import numpy as np
import tempfile
import os

# Optional dependencies
try:
    from metamind.api import (set_api_key, get_api_key,
                              general_image_classifier, ClassificationModel)
except ImportError:
    pass

try:
    from clarifai.client import ClarifaiApi
except ImportError:
    pass

try:
    import cv2
except ImportError:
    pass


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
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
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
        avg_brightness = np.amax(data, 2).mean() / 255.0

        return Value(img, self, {'avg_brightness': avg_brightness})


class SharpnessExtractor(ImageExtractor):

    ''' Gets the degree of blur/sharpness of the image '''

    def __init__(self):
        super(self.__class__, self).__init__()

    def apply(self, img):
        # Taken from
        # http://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry?lq=1
        data = img.data
        gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        sharpness = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray_image, 3))) / 255.0
        return Value(img, self, {'sharpness': sharpness})


class VibranceExtractor(ImageExtractor):

    ''' Gets the variance of color channels of the image '''

    def __init__(self):
        super(self.__class__, self).__init__()

    def apply(self, img):
        data = img.data
        avg_color = np.var(data, 2).mean()
        return Value(img, self, {'avg_color': avg_color})


class MetamindFeaturesExtractor(ImageExtractor):

    ''' Uses the MetaMind API to extract features with an existing classifier.
    Args:
        api_key (str): A valid key for the MetaMind API. Only needs to be
            passed the first time a MetaMindExtractor is initialized.
        classifier (str, int): The name or ID of the MetaMind classifier to
            use. If None or 'general', defaults to the general image
            classifier. Otherwise, must be an integer ID for the desired
            classifier.
    '''

    def __init__(self, api_key=None, classifier=None):
        ImageExtractor.__init__(self)
        api_key = get_api_key() if api_key is None else api_key
        if api_key is None:
            raise ValueError("A valid MetaMind API key must be passed the "
                             "first time a MetaMind extractor is initialized.")
        set_api_key(api_key, verbose=False)

        # TODO: Can add a lookup dictionary somewhere that has name --> ID
        # translation for commonly used classifiers.
        if classifier is None:
            self.classifier = general_image_classifier
        else:
            self.classifier = ClassificationModel(id=classifier)

    def apply(self, img):
        data = img.data
        temp_file = tempfile.mktemp() + '.png'
        cv2.imwrite(temp_file, data)
        labels = self.classifier.predict(temp_file, input_type='files')
        os.remove(temp_file)
        time.sleep(1.0)  # Prevents server error somewhat

        return Value(img, self, {'labels': labels})


class ClarifaiFeaturesExtractor(ImageExtractor):

    ''' Uses the Clarifai API to extract tags of images.
    Args:
        app_id (str): A valid APP_ID for the Clarifai API. Only needs to be
            passed the first time the extractor is initialized.
        app_secret (str): A valid APP_SECRET for the Clarifai API. 
            Only needs to be passed the first time the extractor is initialized.
        model (str): The name of the Clarifai model to use. 
            If None, defaults to the general image tagger. 
    '''

    def __init__(self, app_id=None, app_secret=None, model=None):
        ImageExtractor.__init__(self)
        if app_id is None or app_secret is None:
            try:
                app_id = os.environ['CLARIFAI_APP_ID']
                app_secret = os.environ['CLARIFAI_APP_SECRET']
            except KeyError:
                raise ValueError("A valid Clarifai API APP_ID and APP_SECRET"
                                 "must be passed the first time a Clarifai "
                                 "extractor is initialized.")

        self.tagger = ClarifaiApi(app_id=app_id, app_secret=app_secret)
        if not (model is None):
            self.tagger.set_model(model)

    def apply(self, img):
        data = img.data
        temp_file = tempfile.mktemp() + '.png'
        cv2.imwrite(temp_file, data)
        tags = self.tagger.tag_images(open(temp_file, 'rb'))
        os.remove(temp_file)

        return Value(img, self, {'tags': tags})
