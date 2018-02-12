import os
import requests
from pliers.transformers import Transformer
from pliers.utils import EnvironmentKeyMixin


BASE_URL = 'https://{location}.api.cognitive.microsoft.com/{api}/{version}'\
           '/{method}'


class MicrosoftAPITransformer(Transformer):
    ''' Base MicrosoftAPITransformer class.

    Args:
        subscription_key (str): a valid subscription key for Microsoft Cognitive
            Services. Only needs to be passed the first time the extractor is
            initialized.
        location (str): region the subscription key has been registered in.
            It will be the first part of the endpoint URL suggested by
            Microsoft when you first created the key.
            Examples include: westus, westcentralus, eastus
        api_version (str): API version to use.
    '''

    _log_attributes = ('api_version',)

    def __init__(self, subscription_key=None, location=None,
                 api_version='v1.0'):
        if subscription_key is None:
            if self._env_keys not in os.environ:
                raise ValueError("No Microsoft Cognitive Services credentials "
                                 "found. A Microsoft Azure cognitive service "
                                 "account key must be either passed as the "
                                 "subscription_key argument, or set in the "
                                 "appropriate environment variable.")
            subscription_key = os.environ[self._env_keys]

        if location is None:
            if 'MICROSOFT_SUBSCRIPTION_LOCATION' not in os.environ:
                raise ValueError("No Microsoft Cognitive Services credential "
                                 "location found. The verified region for the "
                                 "provided credentials must be either passed "
                                 "as the location argument, or set in the "
                                 "MICROSOFT_SUBSCRIPTION_LOCATION environment "
                                 "variable.")
            location = os.environ['MICROSOFT_SUBSCRIPTION_LOCATION']

        self.subscription_key = subscription_key
        self.location = location
        self.api_version = api_version
        super(MicrosoftAPITransformer, self).__init__()

    def _query_api(self, stim, params):
        with stim.get_filename() as filename:
            with open(filename, 'rb') as fp:
                data = fp.read()

        headers = {
            'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': self.subscription_key,
        }

        url = BASE_URL.format(location=self.location,
                              api=self.api_name,
                              version=self.api_version,
                              method=self.api_method)

        response = requests.post(url=url,
                                 headers=headers,
                                 params=params,
                                 data=data)
        response = response.json()
        if 'error' in response:
            raise Exception(response['error']['message'])
        elif 'statusCode' in response and response['statusCode'] in [401, 429]:
            raise Exception(response['message'])
        elif 'code' in response and \
             response['code'] == 'NotSupportedVisualFeature':
            raise Exception(response['message'])

        return response


class MicrosoftVisionAPITransformer(MicrosoftAPITransformer, EnvironmentKeyMixin):

    api_name = 'vision'
    _env_keys = 'MICROSOFT_VISION_SUBSCRIPTION_KEY'
