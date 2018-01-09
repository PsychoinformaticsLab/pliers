'''
Extractors that interact with Microsoft Azure Cognitive Services API.
'''

from pliers.extractors.base import ExtractorResult
from pliers.extractors.image import ImageExtractor
from pliers.transformers import MicrosoftAPITransformer
from pliers.utils import attempt_to_import, verify_dependencies


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

        return ExtractorResult([[len(result)]], stim, self, features=['num_faces'])
