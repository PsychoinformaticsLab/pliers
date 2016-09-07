'''
Extractors that operate primarily or exclusively on Image stimuli.
'''

from featurex.stimuli import video
from featurex.extractors import StimExtractor
from featurex.core import Value
import time
import numpy as np
import tempfile
import os
from warnings import warn

# Optional dependencies
try:
    import cv2
except ImportError:
    


class ImageExtractor(StimExtractor):

    ''' Base Image Extractor class; all subclasses can only be applied to
    images. '''
    target = video.ImageStim


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
