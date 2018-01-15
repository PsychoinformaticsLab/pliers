'''
Extractors that interact with Microsoft Azure Cognitive Services API.
'''

from pliers.extractors.base import ExtractorResult
from pliers.extractors.image import ImageExtractor
from pliers.transformers import MicrosoftAPITransformer

import pandas as pd


class MicrosoftAPIFaceExtractor(MicrosoftAPITransformer, ImageExtractor):
    ''' Base MicrosoftAPITransformer class.

    Args:
        face_id (bool): return faceIds of the detected faces or not. The
            default value is true.
        landmarks (str): return face landmarks of the detected faces or
            not. The default value is false.
        attributes (str): analyze and return the one or more specified
            face attributes in the comma-separated string like
            "age,gender". Supported face attributes include age, gender,
            headPose, smile, facialHair, glasses and emotion.
            Note that each face attribute analysis has additional
            computational and time cost.
    '''

    api_name = 'face'
    api_method = 'detect'
    _env_keys = 'MICROSOFT_FACE_SUBSCRIPTION_KEY'
    _log_attributes = ('api_version', 'face_id', 'landmarks', 'attributes')

    def __init__(self, face_id=True, landmarks=False, attributes='', **kwargs):
        self.face_id = face_id
        self.landmarks = landmarks
        self.attributes = attributes
        super(MicrosoftAPIFaceExtractor, self).__init__(**kwargs)

    def _extract(self, stim):
        with stim.get_filename() as filename:
            data = open(filename, 'rb').read()

        params = {
            'returnFaceId': self.face_id,
            'returnFaceLandmarks': self.landmarks,
            'returnFaceAttributes': self.attributes
        }
        raw = self._query_api(data, params)
        return ExtractorResult(raw, stim, self, raw=raw)

    def to_df(self, result):
        cols = []
        data = []
        for i, face in enumerate(result.raw):
            data_dict = {}
            for field, val in face.items():
                if field == 'faceRectangle':
                    lm_pos = 'rectangle_%s'
                    lm_pos = {lm_pos %
                              k: v for (k, v) in val.items()}
                    data_dict.update(lm_pos)
                elif field == 'faceLandmarks':
                    for name, pos in val.items():
                        lm_pos = 'landmark_' + name + '_%s'
                        lm_pos = {lm_pos %
                                  k: v for (k, v) in pos.items()}
                        data_dict.update(lm_pos)
                else:
                    data_dict[field] = val

            names = ['face%d_%s' % (i+1, n) for n in data_dict.keys()]
            cols += names
            data += list(data_dict.values())
        return pd.DataFrame([data], columns=cols)
