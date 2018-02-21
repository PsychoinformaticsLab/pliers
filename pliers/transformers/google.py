import base64
import os
from pliers.transformers import Transformer, BatchTransformerMixin
from pliers.utils import (EnvironmentKeyMixin, attempt_to_import,
                          verify_dependencies)


googleapiclient = attempt_to_import('googleapiclient', fromlist=['discovery'])
oauth_client = attempt_to_import('oauth2client.client', 'oauth_client',
                                 ['GoogleCredentials'])


DISCOVERY_URL = 'https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'


class GoogleAPITransformer(Transformer, EnvironmentKeyMixin):
    ''' Base GoogleAPITransformer class.

    Args:
      discovery_file (str): path to discovery file containing Google
        application credentials.
      api_version (str): API version to use.
      max_results (int): Max number of results per page.
      num_retries (int): Number of times to retry query on failure.
    '''

    _env_keys = 'GOOGLE_APPLICATION_CREDENTIALS'
    _log_attributes = ('api_version',)

    def __init__(self, discovery_file=None, api_version='v1', max_results=100,
                 num_retries=3):
        verify_dependencies(['googleapiclient', 'oauth_client'])
        if discovery_file is None:
            if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
                raise ValueError("No Google application credentials found. "
                                 "A JSON service account key must be either "
                                 "passed as the discovery_file argument, or "
                                 "set in the GOOGLE_APPLICATION_CREDENTIALS "
                                 "environment variable.")
            discovery_file = os.environ['GOOGLE_APPLICATION_CREDENTIALS']

        self.credentials = oauth_client.GoogleCredentials.from_stream(
            discovery_file)
        self.max_results = max_results
        self.num_retries = num_retries
        self.api_version = api_version
        self.service = googleapiclient.discovery.build(
            self.api_name, self.api_version, credentials=self.credentials,
            discoveryServiceUrl=DISCOVERY_URL)
        super(GoogleAPITransformer, self).__init__()


class GoogleVisionAPITransformer(BatchTransformerMixin, GoogleAPITransformer):

    api_name = 'vision'
    _batch_size = 10

    def _query_api(self, request):
        request_obj = self.service.images() \
            .annotate(body={'requests': request})
        return request_obj.execute(num_retries=self.num_retries)['responses']

    def _build_request(self, stims):
        request = []
        for image in stims:
            with image.get_filename() as filename:
                with open(filename, 'rb') as f:
                    img_data = f.read()

            content = base64.b64encode(img_data).decode()
            request.append(
                {
                    'image': {'content': content},
                    'features': [{
                        'type': self.request_type,
                        'maxResults': self.max_results,
                    }]
                })

        return request
