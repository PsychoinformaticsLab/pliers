import logging
import os
from pliers.transformers import BatchTransformerMixin
from pliers.transformers.api import APITransformer
from pliers.utils import attempt_to_import, verify_dependencies


googleapiclient = attempt_to_import('googleapiclient', fromlist=['discovery'])
google_auth = attempt_to_import('google.oauth2', 'google_auth',
                                fromlist=['service_account'])


DISCOVERY_URL = 'https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'


class GoogleAPITransformer(APITransformer):
    ''' Base GoogleAPITransformer class.

    Args:
      discovery_file (str): path to discovery file containing Google
        application credentials.
      api_version (str): API version to use.
      max_results (int): Max number of results per page.
      num_retries (int): Number of times to retry query on failure.
      rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
    '''

    _env_keys = 'GOOGLE_APPLICATION_CREDENTIALS'
    _log_attributes = ('discovery_file', 'api_version')

    def __init__(self, discovery_file=None, api_version='v1', max_results=100,
                 num_retries=3, rate_limit=None, **kwargs):
        verify_dependencies(['googleapiclient', 'google_auth'])
        if discovery_file is None:
            if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
                raise ValueError("No Google application credentials found. "
                                 "A JSON service account key must be either "
                                 "passed as the discovery_file argument, or "
                                 "set in the GOOGLE_APPLICATION_CREDENTIALS "
                                 "environment variable.")
            discovery_file = os.environ['GOOGLE_APPLICATION_CREDENTIALS']

        self.discovery_file = discovery_file
        try:
            self.credentials = google_auth.service_account.Credentials\
                .from_service_account_file(discovery_file)
            self.service = googleapiclient.discovery.build(
                self.api_name, api_version, credentials=self.credentials,
                discoveryServiceUrl=DISCOVERY_URL)
        except Exception as e:
            logging.warn(str(e))
            self.credentials = None
            self.service = None
        self.max_results = max_results
        self.num_retries = num_retries
        self.api_version = api_version
        super(GoogleAPITransformer, self).__init__(rate_limit=rate_limit,
                                                   **kwargs)

    @property
    def api_keys(self):
        return [self.credentials]

    def check_valid_keys(self):
        return self.credentials is not None


class GoogleVisionAPITransformer(GoogleAPITransformer, BatchTransformerMixin):

    ''' Base class for transformers using the Google Vision API.

    Args:
        discovery_file (str): path to discovery file containing Google
            application credentials.
        api_version (str): API version to use.
        max_results (int): Max number of results per page.
        num_retries (int): Number of times to retry query on failure.
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
        batch_size (int): Number of stims to send per batched API request.
    '''

    api_name = 'vision'
    _batch_size = 1

    def __init__(self, discovery_file=None, api_version='v1', max_results=100,
                 num_retries=3, rate_limit=None, batch_size=None):
        super(GoogleVisionAPITransformer, self).__init__(discovery_file=discovery_file,
                                                         api_version=api_version,
                                                         max_results=max_results,
                                                         num_retries=num_retries,
                                                         rate_limit=rate_limit,
                                                         batch_size=batch_size)

    def _query_api(self, request):
        request_obj = self.service.images() \
            .annotate(body={'requests': request})
        return request_obj.execute(num_retries=self.num_retries)['responses']

    def _build_request(self, stims):
        request = []
        for image in stims:
            image_desc = {}
            if image.url:
                image_desc['source'] = {
                    'imageUri': image.url
                }
            else:
                image_desc['content'] = image.get_bytestring()

            request.append(
                {
                    'image': image_desc,
                    'features': [{
                        'type': self.request_type,
                        'maxResults': self.max_results,
                    }]
                })

        return request
