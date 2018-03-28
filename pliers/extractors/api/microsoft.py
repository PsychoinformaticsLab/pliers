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
        face_id (bool): Return faceIds of the detected faces or not. The
            default value is False.
        landmarks (str): Return face landmarks of the detected faces or
            not. The default value is False.
        attributes (list): One or more specified face attributes as strings.
            Supported face attributes include accessories, age, blur, emotion,
            exposure, facialHair, gender, glasses, hair, headPose, makeup,
            noise, occlusion, and smile. Note that each attribute has
            additional computational and time cost.
        subscription_key (str): A valid subscription key for Microsoft Cognitive
            Services. Only needs to be passed the first time the extractor is
            initialized.
        location (str): Region the subscription key has been registered in.
            It will be the first part of the endpoint URL suggested by
            Microsoft when you first created the key.
            Examples include: westus, westcentralus, eastus
        api_version (str): API version to use.
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
    '''

    api_name = 'face'
    api_method = 'detect'
    _env_keys = 'MICROSOFT_FACE_SUBSCRIPTION_KEY'
    _log_attributes = ('subscription_key', 'location', 'api_version',
                       'face_id', 'rectangle', 'landmarks', 'attributes')

    def __init__(self, face_id=False, rectangle=True, landmarks=False,
                 attributes=None, subscription_key=None, location=None,
                 api_version='v1.0', rate_limit=None):
        self.face_id = face_id
        self.rectangle = rectangle
        self.landmarks = landmarks
        self.attributes = attributes
        super(MicrosoftAPIFaceExtractor,
              self).__init__(subscription_key=subscription_key,
                             location=location,
                             api_version=api_version,
                             rate_limit=rate_limit)

    def _extract(self, stim):
        if self.attributes:
            attributes = ','.join(self.attributes)
        else:
            attributes = ''

        params = {
            'returnFaceId': self.face_id,
            'returnFaceLandmarks': self.landmarks,
            'returnFaceAttributes': attributes
        }
        raw = self._query_api(stim, params)
        return ExtractorResult(raw, stim, self)

    def _parse_response_json(self, json):
        data_dict = {}
        for k, v in json.items():
            if k == 'faceRectangle' and not self.rectangle:
                continue
            if k == 'faceAttributes':
                k = 'face'
            if isinstance(v, dict):
                subdata = self._parse_response_json(v)
                for sk, sv in subdata.items():
                    data_dict['%s_%s' % (k, sk)] = sv
            elif isinstance(v, list):
                # Hard coded to this extractor
                for attr in v:
                    if k == 'hairColor':
                        key = attr['color']
                    elif k == 'accessories':
                        key = '%s_%s' % (k, attr['type'])
                    else:
                        continue
                    data_dict[key] = attr['confidence']
            else:
                data_dict[k] = v
        return data_dict

    def _to_df(self, result):
        face_results = []
        for i, face in enumerate(result._data):
            face_data = self._parse_response_json(face)
            face_results.append(face_data)

        return pd.DataFrame(face_results)


class MicrosoftAPIFaceEmotionExtractor(MicrosoftAPIFaceExtractor):

    ''' Extracts facial emotions from images using the Microsoft API '''

    def __init__(self, face_id=False, rectangle=False, landmarks=False,
                 subscription_key=None, location=None, api_version='v1.0',
                 rate_limit=None):
        super(MicrosoftAPIFaceEmotionExtractor,
              self).__init__(face_id,
                             rectangle,
                             landmarks,
                             ['emotion'],
                             subscription_key=subscription_key,
                             location=location,
                             api_version=api_version,
                             rate_limit=rate_limit)


class MicrosoftVisionAPIExtractor(MicrosoftVisionAPITransformer,
                                  ImageExtractor):
    ''' Base MicrosoftVisionAPIExtractor class.

    Args:
        features (list): One or more specified vision features as strings.
            Supported vision features include Tags, Categories, ImageType,
            Color, and Adult. Note that each attribute has additional
            computational and time cost. By default extracts all visual
            features from an image.
        subscription_key (str): A valid subscription key for Microsoft Cognitive
            Services. Only needs to be passed the first time the extractor is
            initialized.
        location (str): Region the subscription key has been registered in.
            It will be the first part of the endpoint URL suggested by
            Microsoft when you first created the key.
            Examples include: westus, westcentralus, eastus
        api_version (str): API version to use.
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
    '''

    api_method = 'analyze'
    _log_attributes = ('subscription_key', 'location', 'api_version',
                       'features')

    def __init__(self, features=None, subscription_key=None, location=None,
                 api_version='v1.0', rate_limit=None):
        self.features = features if features else ['Tags', 'Categories',
                                                   'ImageType', 'Color',
                                                   'Adult']
        super(MicrosoftVisionAPIExtractor,
              self).__init__(subscription_key=subscription_key,
                             location=location,
                             api_version=api_version,
                             rate_limit=rate_limit)

    def _extract(self, stim):
        params = {
            'visualFeatures': ','.join(self.features),
        }
        raw = self._query_api(stim, params)
        return ExtractorResult(raw, stim, self)

    def _to_df(self, result):
        data_dict = {}
        for feat in self.features:
            feat = feat[0].lower() + feat[1:]
            if feat in result._data:
                if feat == 'tags':
                    for tag in result._data[feat]:
                        data_dict[tag['name']] = tag['confidence']
                elif feat == 'categories':
                    for cat in result._data[feat]:
                        data_dict[cat['name']] = cat['score']
                else:
                    data_dict.update(result._data[feat])
        return pd.DataFrame([data_dict.values()], columns=data_dict.keys())


class MicrosoftVisionAPITagExtractor(MicrosoftVisionAPIExtractor):

    ''' Extracts image tags using the Microsoft API '''

    def __init__(self, subscription_key=None, location=None, api_version='v1.0',
                 rate_limit=None):
        super(MicrosoftVisionAPITagExtractor,
              self).__init__(features=['Tags'],
                             subscription_key=subscription_key,
                             location=location,
                             api_version=api_version,
                             rate_limit=rate_limit)


class MicrosoftVisionAPICategoryExtractor(MicrosoftVisionAPIExtractor):

    ''' Extracts image categories using the Microsoft API '''

    def __init__(self, subscription_key=None, location=None, api_version='v1.0',
                 rate_limit=None):
        super(MicrosoftVisionAPICategoryExtractor,
              self).__init__(features=['Categories'],
                             subscription_key=subscription_key,
                             location=location,
                             api_version=api_version,
                             rate_limit=rate_limit)


class MicrosoftVisionAPIImageTypeExtractor(MicrosoftVisionAPIExtractor):

    ''' Extracts image types (clipart, etc.) using the Microsoft API '''

    def __init__(self, subscription_key=None, location=None, api_version='v1.0',
                 rate_limit=None):
        super(MicrosoftVisionAPIImageTypeExtractor,
              self).__init__(features=['ImageType'],
                             subscription_key=subscription_key,
                             location=location,
                             api_version=api_version,
                             rate_limit=rate_limit)


class MicrosoftVisionAPIColorExtractor(MicrosoftVisionAPIExtractor):

    ''' Extracts image color attributes using the Microsoft API '''

    def __init__(self, subscription_key=None, location=None, api_version='v1.0',
                 rate_limit=None):
        super(MicrosoftVisionAPIColorExtractor,
              self).__init__(features=['Color'],
                             subscription_key=subscription_key,
                             location=location,
                             api_version=api_version,
                             rate_limit=rate_limit)


class MicrosoftVisionAPIAdultExtractor(MicrosoftVisionAPIExtractor):

    ''' Extracts the presence of adult content using the Microsoft API '''

    def __init__(self, subscription_key=None, location=None, api_version='v1.0',
                 rate_limit=None):
        super(MicrosoftVisionAPIAdultExtractor,
              self).__init__(features=['Adult'],
                             subscription_key=subscription_key,
                             location=location,
                             api_version=api_version,
                             rate_limit=rate_limit)
