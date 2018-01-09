import os
from pliers.transformers import Transformer
from pliers.utils import EnvironmentKeyMixin


BASE_URL = 'https://{location}.api.cognitive.microsoft.com/{api}/{version}/'


class MicrosoftAPITransformer(Transformer):
    ''' Base MicrosoftAPITransformer class.

    Args:
        subscription_key (str): a valid subscription key for Microsoft Cognitive
            Services. Only needs to be passed the first time the extractor is
            initialized.
        location (str): region the subscription key has been registered in.
        api_version (str): API version to use.
    '''

    _log_attributes = ('api_version',)

    def __init__(self, subscription_key=None, location='westus',
                 api_version='v1.0'):
        if subscription_key is None:
            if self._env_keys not in os.environ:
                raise ValueError("No Microsoft Cognitive Services credentials "
                                 "found. A Microsoft Azure cognitive service "
                                 "account key must be either passed as the "
                                 "subscription_key argument, or set in the "
                                 "appropriate environment variable.")
            subscription_key = os.environ[self._env_keys]

        self.subscription_key = subscription_key
        self.location = location
        self.api_version = api_version
        self.base_url = BASE_URL.format(location=location,
                                        api=self.api_name,
                                        version=api_version)
        super(MicrosoftAPITransformer, self).__init__()


class MicrosoftVisionAPITransformer(MicrosoftAPITransformer, EnvironmentKeyMixin):

    api_name = 'vision'
    _env_keys = 'MICROSOFT_VISION_SUBSCRIPTION_KEY'

    def _query_api(self, request):
        pass

    def _build_request(self, stims):
        pass
