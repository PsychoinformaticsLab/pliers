'''
Extractors that operate primarily or exclusively on Image stimuli.
'''

from pliers.stimuli.image import ImageStim
from pliers.extractors.base import Extractor, ExtractorResult
import numpy as np


class ImageExtractor(Extractor):

    ''' Base Image Extractor class; all subclasses can only be applied to
    images. '''
    _input_type = ImageStim


class BrightnessExtractor(ImageExtractor):

    ''' Gets the average luminosity of the pixels in the image '''

    def __init__(self):
        super(self.__class__, self).__init__()

    def _extract(self, stim):
        data = stim.data
        brightness = np.amax(data, 2).mean() / 255.0

        return ExtractorResult(np.array([[brightness]]), stim, self,
                               features=['brightness'])


class SharpnessExtractor(ImageExtractor):

    ''' Gets the degree of blur/sharpness of the image '''

    def __init__(self):
        super(self.__class__, self).__init__()

    def _extract(self, stim):
        import cv2
        # Taken from
        # http://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry?lq=1
        data = stim.data
        gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        sharpness = np.max(
            cv2.convertScaleAbs(cv2.Laplacian(gray_image, 3))) / 255.0
        return ExtractorResult(np.array([[sharpness]]), stim, self,
                               features=['sharpness'])


class VibranceExtractor(ImageExtractor):

    ''' Gets the variance of color channels of the image '''

    def __init__(self):
        super(self.__class__, self).__init__()

    def _extract(self, stim):
        data = stim.data
        vibrance = np.var(data, 2).mean()
        return ExtractorResult(np.array([[vibrance]]), stim, self,
                               features=['vibrance'])


class SaliencyExtractor(ImageExtractor):

    ''' Determines the saliency of the image using Itti & Koch (1998) algorithm
    implemented in pySaliencyMap '''

    def __init__(self):
        super(self.__class__, self).__init__()

    def _extract(self, stim):
        from pliers.external.pysaliency import pySaliencyMap
        # pySaliencyMap from https://github.com/akisato-/pySaliencyMap
        # Initialize variables
        h, w, c = stim.data.shape
        sm = pySaliencyMap.pySaliencyMap(h, w)

        # Compute saliency maps and store full maps as derivatives
        stim.derivatives = dict()
        stim.derivatives['saliency_map'] = sm.SMGetSM(stim.data)
        stim.derivatives['binarized_map'] = sm.SMGetBinarizedSM(
            stim.data)  # thresholding done using Otsu

        # Compute summary statistics
        output = {}
        output['max_saliency'] = np.max(stim.derivatives['saliency_map'])
        output['max_y'], output['max_x'] = [list(i)[0] for i in np.where(
            stim.derivatives['saliency_map'] == output['max_saliency'])]
        output['frac_high_saliency'] = np.sum(
            stim.derivatives['binarized_map']/255.0)/(h * w)

        return ExtractorResult(np.array([list(output.values())]), stim, self,
                               features=list(output.keys()))
