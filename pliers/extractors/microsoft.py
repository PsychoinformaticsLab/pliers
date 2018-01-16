'''
Extractors that interact with Microsoft Azure Cognitive Services API.
'''

from pliers.extractors.base import ExtractorResult
from pliers.extractors.image import ImageExtractor
from pliers.transformers import (MicrosoftAPITransformer,
                                 MicrosoftVisionAPITransformer)

import pandas as pd


class MicrosoftAPIFaceExtractor(MicrosoftAPITransformer, ImageExtractor):
    ''' Base MicrosoftAPITransformer class.

    Args:
        face_id (bool): return faceIds of the detected faces or not. The
            default value is False.
        landmarks (str): return face landmarks of the detected faces or
            not. The default value is False.
        attributes (list): one or more specified face attributes as strings.
            Supported face attributes include age, gender, headPose, smile,
            facialHair, glasses and emotion. Note that each attribute has
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

        params = {
            'returnFaceId': self.face_id,
            'returnFaceLandmarks': self.landmarks,
            'returnFaceAttributes': attributes
        }
        raw = self._query_api(data, params)
        return ExtractorResult(raw, stim, self, raw=raw)

    def to_df(self, result):
        cols = []
        data = []

        def update_with_prefix(primary_dict, updating_dict, prefix):
            update_dict = {prefix + k: v for (k, v) in updating_dict.items()}
            primary_dict.update(update_dict)

        for i, face in enumerate(result.raw):
            data_dict = {}
            for field, val in face.items():
                if field == 'faceRectangle':
                    update_with_prefix(data_dict,
                                       val,
                                       'rectangle_')
                elif field == 'faceLandmarks':
                    for name, pos in val.items():
                        update_with_prefix(data_dict,
                                           pos,
                                           'landmark_%s_' % (name))
                elif field == 'faceAttributes':
                    attributes = val.items()
                    for k, v in attributes:
                        if k == 'accessories':
                            for accessory in v:
                                name = 'accessory_' + accessory['type']
                                data_dict[name] = accessory['confidence']
                        elif k == 'hair':
                            update_with_prefix(data_dict,
                                               {'bald': v['bald'],
                                                'invisible': v['invisible']},
                                               'hair_')
                            for color in v['hairColor']:
                                feature_name = 'hairColor_%s' % color['color']
                                data_dict[feature_name] = color['confidence']
                        elif isinstance(v, dict):
                            update_with_prefix(data_dict,
                                               v,
                                               '%s_' % (k))
                        else:
                            data_dict[k] = v
                else:
                    data_dict[field] = val

            names = ['face%d_%s' % (i+1, n) for n in data_dict.keys()]
            cols += names
            data += list(data_dict.values())
        return pd.DataFrame([data], columns=cols)


class MicrosoftVisionAPIExtractor(MicrosoftVisionAPITransformer,
                                  ImageExtractor):

    def _extract(self, stim):
        pass
