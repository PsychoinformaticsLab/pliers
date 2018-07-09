import logging
import os
import requests
from pliers.transformers.api import APITransformer


BASE_URL = 'https://{location}.api.cognitive.microsoft.com/{api}/{version}'\
           '/{method}'


class MicrosoftAPITransformer(APITransformer):
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
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
    '''

    _log_attributes = ('subscription_key', 'location', 'api_version')
    _rate_limit = 3

    def __init__(self, subscription_key=None, location=None,
                 api_version='v1.0', rate_limit=None):
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
        super(MicrosoftAPITransformer, self).__init__(rate_limit=rate_limit)

    @property
    def api_keys(self):
        return [self.subscription_key]

    def check_valid_keys(self):
        try:
            headers = {
                'Content-Type': 'application/octet-stream',
                'Ocp-Apim-Subscription-Key': self.subscription_key
            }
            self._send_request('', headers=headers, params={})
            return True
        except Exception as e:
            if 'too small' in str(e):
                return True
            elif 'invalid subscription' in str(e):
                logging.warn(str(e))
                return False
            elif '[Errno 8]' in str(e):
                logging.warn(str(e))
                return False
            else:
                raise e

    def _query_api(self, stim, params):
        headers = {
            'Ocp-Apim-Subscription-Key': self.subscription_key
        }
        if stim.url:
            headers['Content-Type'] = 'application/json'
            data = None
            json = {'url': stim.url}
        else:
            headers['Content-Type'] = 'application/octet-stream'
            with stim.get_filename() as filename:
                with open(filename, 'rb') as fp:
                    data = fp.read()
            json = None

        return self._send_request(data=data, json=json, headers=headers,
                                  params=params)

    def _send_request(self, data=None, json=None, headers=None, params=None):
        url = BASE_URL.format(location=self.location,
                              api=self.api_name,
                              version=self.api_version,
                              method=self.api_method)

        response = requests.post(url=url,
                                 headers=headers,
                                 params=params,
                                 data=data,
                                 json=json)
        response = response.json()
        if 'error' in response:
            raise Exception(response['error']['message'])
        elif 'statusCode' in response and response['statusCode'] in [401, 429]:
            raise Exception(response['message'])
        elif 'code' in response and \
             response['code'] == 'NotSupportedVisualFeature':
            raise Exception(response['message'])

        return response


class MicrosoftVisionAPITransformer(MicrosoftAPITransformer):

    api_name = 'vision'
    _env_keys = 'MICROSOFT_VISION_SUBSCRIPTION_KEY'
