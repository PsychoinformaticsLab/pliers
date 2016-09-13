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
    pass

try:
    import pytesseract
except ImportError:
    pass

try:
    import Image
except ImportError:
    from PIL import Image

class ImageExtractor(StimExtractor):

    ''' Base Image Extractor class; all subclasses can only be applied to
    images. '''
    target = video.ImageStim


class BrightnessExtractor(ImageExtractor):

    ''' Gets the average luminosity of the pixels in the image '''

    def __init__(self):
        super(self.__class__, self).__init__()

    def apply(self, stim):
        data = stim.data
        avg_brightness = np.amax(data, 2).mean() / 255.0

        return Value(stim, self, {'avg_brightness': avg_brightness})


class SharpnessExtractor(ImageExtractor):

    ''' Gets the degree of blur/sharpness of the image '''

    def __init__(self):
        super(self.__class__, self).__init__()

    def apply(self, stim):
        # Taken from
        # http://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry?lq=1
        data = stim.data
        gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        sharpness = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray_image, 3))) / 255.0
        return Value(stim, self, {'sharpness': sharpness})


class VibranceExtractor(ImageExtractor):

    ''' Gets the variance of color channels of the image '''

    def __init__(self):
        super(self.__class__, self).__init__()

    def apply(self, stim):
        data = stim.data
        avg_color = np.var(data, 2).mean()
        return Value(stim, self, {'avg_color': avg_color})


class TesseractExtractor(ImageExtractor):

    ''' Uses the Tesseract library to extract text from images '''

    def __init__(self):
        ImageExtractor.__init__(self)

    def apply(self, img):
        data = img.data
        text = pytesseract.image_to_string(Image.fromarray(data))

        return Value(img, self, {'text': text})


class SaliencyExtractor(ImageExtractor):

    ''' Determines the saliency of the image using Itti & Koch (1998) algorithm implemented in pySaliencyMap '''

    def __init__(self):
        super(self.__class__, self).__init__()

    def apply(self, stim):
        from featurex.external import pySaliencyMap
        # pySaliencyMap from https://github.com/akisato-/pySaliencyMap
        data = stim.data

        # Initialize variables
        h, w, c = stim.data.shape
        sm = pySaliencyMap.pySaliencyMap(h, w)

        # Compute saliency maps and store full maps as derivatives
        stim.derivatives = dict()
        stim.derivatives['saliency_map'] = sm.SMGetSM(stim.data)
        stim.derivatives['binarized_map'] = sm.SMGetBinarizedSM(stim.data) #thresholding done using Otsu

        # Compute summary statistics
        output = {}
        output['max_saliency'] = np.max(stim.derivatives['saliency_map'])
        output['max_y'], output['max_x'] = [list(i)[0] for i in np.where(stim.derivatives['saliency_map']==output['max_saliency'])]
        output['frac_high_saliency'] = np.sum(stim.derivatives['binarized_map']/255.0)/(h * w)

        return Value(stim, self, output)
