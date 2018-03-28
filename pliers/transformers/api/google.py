import base64
import os
from pliers.transformers import BatchTransformerMixin
from pliers.transformers.api import APITransformer
from pliers.utils import attempt_to_import, verify_dependencies


googleapiclient = attempt_to_import('googleapiclient', fromlist=['discovery'])
oauth_client = attempt_to_import('oauth2client.client', 'oauth_client',
                                 ['GoogleCredentials'])


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
                 num_retries=3, rate_limit=None):
        verify_dependencies(['googleapiclient', 'oauth_client'])
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
            self.credentials = oauth_client.GoogleCredentials.from_stream(
                discovery_file)
            self.service = googleapiclient.discovery.build(
                self.api_name, api_version, credentials=self.credentials,
                discoveryServiceUrl=DISCOVERY_URL)
        except:
            self.credentials = None
            self.service = None
        self.max_results = max_results
        self.num_retries = num_retries
        self.api_version = api_version
        super(GoogleAPITransformer, self).__init__(rate_limit=rate_limit)

    @property
    def api_keys(self):
        return [self.credentials]

    def check_valid_keys(self):
        return self.credentials is not None


class GoogleVisionAPITransformer(GoogleAPITransformer, BatchTransformerMixin):

    api_name = 'vision'
    _batch_size = 1

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
                with image.get_filename() as filename:
                    with open(filename, 'rb') as f:
                        img_data = f.read()
                image_desc['content'] = base64.b64encode(img_data).decode()

            request.append(
                {
                    'image': image_desc,
                    'features': [{
                        'type': self.request_type,
                        'maxResults': self.max_results,
                    }]
                })

        return request
