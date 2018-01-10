'''
Extractors that interact with Microsoft Azure Cognitive Services API.
'''

from pliers.extractors.base import ExtractorResult
from pliers.extractors.image import ImageExtractor
from pliers.transformers import MicrosoftAPITransformer
from pliers.utils import attempt_to_import, verify_dependencies

import pandas as pd


CF = attempt_to_import('cognitive_face', 'CF')


class MicrosoftAPIFaceExtractor(MicrosoftAPITransformer, ImageExtractor):

    api_name = 'face'
    _env_keys = 'MICROSOFT_FACE_SUBSCRIPTION_KEY'

    def __init__(self, *args, **kwargs):
        super(MicrosoftAPIFaceExtractor, self).__init__(*args, **kwargs)
        verify_dependencies(['CF'])
        CF.Key.set(self.subscription_key)
        CF.BaseUrl.set(self.base_url)

    def _extract(self, stim):
        with stim.get_filename() as filename:
            result = CF.face.detect(filename)

        return ExtractorResult(result, stim, self, raw=result)

    def to_df(self, result):
        n_faces = len(result.raw)
        cols = ['faceId', 'faceRectangleTop', 'faceRectangleLeft',
                'faceRectangleWidth', 'faceRectangleHeight']
        if n_faces > 1:
            new_cols = []
            for i in range(1, n_faces + 1):
                new_cols.extend(['%s_%d' % (c, i) for c in cols])
            cols = new_cols
        data = []
        for i, face in enumerate(result.raw):
            data.extend([face['faceId'],
                         face['faceRectangle']['top'],
                         face['faceRectangle']['left'],
                         face['faceRectangle']['width'],
                         face['faceRectangle']['height']])
        return pd.DataFrame([data], columns=cols)
