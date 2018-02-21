'''
Extractors that operate primarily or exclusively on Image stimuli.
'''

from pliers.stimuli.image import ImageStim
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.utils import attempt_to_import, verify_dependencies, listify
from pliers.support.due import due, Url, Doi
import numpy as np
import pandas as pd
from functools import partial


cv2 = attempt_to_import('cv2')
face_recognition = attempt_to_import('face_recognition')


class ImageExtractor(Extractor):

    ''' Base Image Extractor class; all subclasses can only be applied to
    images. '''
    _input_type = ImageStim


class BrightnessExtractor(ImageExtractor):

    ''' Gets the average luminosity of the pixels in the image '''

    VERSION = '1.0'

    def _extract(self, stim):
        data = stim.data
        brightness = np.amax(data, 2).mean() / 255.0

        return ExtractorResult(np.array([[brightness]]), stim, self,
                               features=['brightness'])


class SharpnessExtractor(ImageExtractor):

    ''' Gets the degree of blur/sharpness of the image '''

    VERSION = '1.0'

    def _extract(self, stim):
        verify_dependencies(['cv2'])
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

    VERSION = '1.0'

    def _extract(self, stim):
        data = stim.data
        vibrance = np.var(data, 2).mean()
        return ExtractorResult(np.array([[vibrance]]), stim, self,
                               features=['vibrance'])


class SaliencyExtractor(ImageExtractor):

    ''' Determines the saliency of the image using Itti & Koch (1998) algorithm
    implemented in pySaliencyMap '''

    @due.dcite(Doi("10.1109/34.730558"),
               description="Image saliency estimation",
               path='pliers.extractors.image.SilencyExtractor',
               tags=["implementation"])
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


class FaceRecognitionFeatureExtractor(ImageExtractor):

    _log_attributes = ('face_recognition_kwargs',)

    def __init__(self, **face_recognition_kwargs):
        verify_dependencies(['face_recognition'])

        self.face_recognition_kwargs = face_recognition_kwargs
        func = getattr(face_recognition.api, self._feature)
        self.func = partial(func, **face_recognition_kwargs)

        super(FaceRecognitionFeatureExtractor, self).__init__()

    def get_feature_names(self):
        return self._feature

    def _extract(self, stim):
        values = self.func(stim.data)
        feature_names = listify(self.get_feature_names())
        return ExtractorResult(values, stim, self, features=feature_names)

    def _to_df(self, result):
        cols = listify(self._feature)
        return pd.DataFrame([[r] for r in result._data], columns=cols)


class FaceRecognitionFaceEncodingsExtractor(FaceRecognitionFeatureExtractor):
    ''' Uses the face_recognition package to extract a 128-dimensional encoding
    for every face detected in an image. For details, see documentation for
    face_recognition.api.face_encodings. '''

    _feature = 'face_encodings'


class FaceRecognitionFaceLandmarksExtractor(FaceRecognitionFeatureExtractor):
    ''' Uses the face_recognition package to extract the locations of named
    features of faces in the image. For details, see documentation for
    face_recognition.api.face_landmarks.'''

    _feature = 'face_landmarks'

    def _to_df(self, result):
        data = pd.DataFrame.from_records(result._data)
        data.columns = ['%s_%s' % (self._feature, c) for c in data.columns]
        return data


class FaceRecognitionFaceLocationsExtractor(FaceRecognitionFeatureExtractor):
    ''' Uses the face_recognition package to extract bounding boxes for all
    faces in an image. For details, see documentation for
    face_recognition.api.face_locations. '''

    _feature = 'face_locations'
