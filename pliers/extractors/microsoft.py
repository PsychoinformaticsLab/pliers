'''
Extractors that interact with Microsoft Azure Cognitive Services API.
'''

from pliers.extractors.base import ExtractorResult
from pliers.extractors.image import ImageExtractor
from pliers.transformers import (MicrosoftAPITransformer,
                                 MicrosoftVisionAPITransformer)

import pandas as pd


class MicrosoftAPIFaceExtractor(MicrosoftAPITransformer, ImageExtractor):
    ''' Extracts face features (location, emotion, accessories, etc.). From an
    image using the Microsoft Azure Cognitive Services API.

    Args:
        face_id (bool): return faceIds of the detected faces or not. The
            default value is False.
        landmarks (str): return face landmarks of the detected faces or
            not. The default value is False.
        attributes (list): one or more specified face attributes as strings.
            Supported face attributes include accessories, age, blur, emotion,
            exposure, facialHair, gender, glasses, hair, headPose, makeup,
            noise, occlusion, and smile. Note that each attribute has
            additional computational and time cost.
    '''

    api_name = 'face'
    api_method = 'detect'
    _env_keys = 'MICROSOFT_FACE_SUBSCRIPTION_KEY'
    _log_attributes = ('api_version', 'face_id', 'landmarks', 'attributes')

    def __init__(self, face_id=False, landmarks=False, attributes=None, **kwargs):
        self.face_id = face_id
        self.landmarks = landmarks
        self.attributes = attributes
        super(MicrosoftAPIFaceExtractor, self).__init__(**kwargs)

    def _extract(self, stim):
        with stim.get_filename() as filename:
            data = open(filename, 'rb').read()

        if self.attributes:
            attributes = ','.join(self.attributes)
        else:
            attributes = ''

        params = {
            'returnFaceId': self.face_id,
            'returnFaceLandmarks': self.landmarks,
            'returnFaceAttributes': attributes
        }
        raw = self._query_api(data, params)
        return ExtractorResult(None, stim, self, raw=raw)

    def _parse_response_json(self, json):
        keys = []
        values = []
        for k, v in json.items():
            if k == 'faceAttributes':
                k = 'face'
            if isinstance(v, dict):
                subkeys, subvalues = self._parse_response_json(v)
                keys.extend(['%s_%s' % (k, s) for s in subkeys])
                values.extend(subvalues)
            elif isinstance(v, list):
                # Hard coded to this extractor
                for attr in v:
                    if k == 'hairColor':
                        keys.append('%s' % attr['color'])
                    elif k == 'accessories':
                        keys.append('%s_%s' % (k, attr['type']))
                    else:
                        continue
                    values.append(attr['confidence'])
            else:
                keys.append(k)
                values.append(v)
        return keys, values

    def _to_df(self, result):
        cols = []
        data = []
        for i, face in enumerate(result.raw):
            face_keys, face_data = self._parse_response_json(face)
            cols = face_keys if i == 0 else cols
            data.append(face_data)

        return pd.DataFrame(data, columns=cols)


class MicrosoftVisionAPIFaceEmotionExtractor(MicrosoftAPIFaceExtractor):

    ''' Extracts facial emotions from images using the Microsoft API '''

    def __init__(self, face_id=False, landmarks=False, **kwargs):
        super(MicrosoftVisionAPIFaceEmotionExtractor, self).__init__(face_id,
                                                                     landmarks,
                                                                     'emotion',
                                                                     **kwargs)


class MicrosoftVisionAPIExtractor(MicrosoftVisionAPITransformer,
                                  ImageExtractor):
    ''' Base MicrosoftVisionAPIExtractor class. By default extracts all visual
    features from an image.

    Args:
        face_id (bool): return faceIds of the detected faces or not. The
            default value is False.
        landmarks (str): return face landmarks of the detected faces or
            not. The default value is False.
        attributes (list): one or more specified face attributes as strings.
            Supported face attributes include accessories, age, blur, emotion,
            exposure, facialHair, gender, glasses, hair, headPose, makeup,
            noise, occlusion, and smile. Note that each attribute has
            additional computational and time cost.
    '''

    api_method = 'analyze'

    def __init__(self, features='Tags,Categories,ImageType,Color,Adult',
                 **kwargs):
        if hasattr(self, '_feature'):
            self.features = self._feature
        else:
            self.features = features
        super(MicrosoftVisionAPIExtractor, self).__init__(**kwargs)

    def _extract(self, stim):
        with stim.get_filename() as filename:
            data = open(filename, 'rb').read()

        params = {
            'visualFeatures': self.features,
        }
        raw = self._query_api(data, params)
        return ExtractorResult(None, stim, self, raw=raw)

    def _to_df(self, result):
        features = self.features.split(',')

        data_dict = {}
        for feat in features:
            feat = feat[0].lower() + feat[1:]
            if feat == 'tags':
                for tag in result.raw[feat]:
                    data_dict[tag['name']] = tag['confidence']
            elif feat == 'categories':
                for cat in result.raw[feat]:
                    data_dict[cat['name']] = cat['score']
            else:
                data_dict.update(result.raw[feat])
        return pd.DataFrame([data_dict.values()], columns=data_dict.keys())


class MicrosoftVisionAPITagExtractor(MicrosoftVisionAPIExtractor):

    ''' Extracts image tags using the Microsoft API '''

    _feature = 'Tags'


class MicrosoftVisionAPICategoryExtractor(MicrosoftVisionAPIExtractor):

    ''' Extracts image categories using the Microsoft API '''

    _feature = 'Categories'


class MicrosoftVisionAPIImageTypeExtractor(MicrosoftVisionAPIExtractor):

    ''' Extracts image types (clipart, etc.) using the Microsoft API '''

    _feature = 'ImageType'


class MicrosoftVisionAPIColorExtractor(MicrosoftVisionAPIExtractor):

    ''' Extracts image color attributes using the Microsoft API '''

    _feature = 'Color'


class MicrosoftVisionAPIAdultExtractor(MicrosoftVisionAPIExtractor):

    ''' Extracts the presence of adult content using the Microsoft API '''

    _feature = 'Adult'
