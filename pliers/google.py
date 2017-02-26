import base64
import os
import tempfile
from scipy.misc import imsave
from pliers.transformers import (Transformer, BatchTransformerMixin,
                                 EnvironmentKeyMixin)

try:
    from googleapiclient import discovery
    from oauth2client.client import GoogleCredentials
except ImportError:
    pass


DISCOVERY_URL = 'https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'
BATCH_SIZE = 10


class GoogleAPITransformer(Transformer, BatchTransformerMixin, EnvironmentKeyMixin):

    _env_keys = 'GOOGLE_APPLICATION_CREDENTIALS'
    _log_attributes = ('handle_annotations',)

    def __init__(self, discovery_file=None, api_version='v1', max_results=100,
                 num_retries=3, handle_annotations='prefix'):

        if discovery_file is None:
            if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
                raise ValueError("No Google application credentials found. "
                                 "A JSON service account key must be either "
                                 "passed as the discovery_file argument, or "
                                 "set in the GOOGLE_APPLICATION_CREDENTIALS "
                                 "environment variable.")
            discovery_file = os.environ['GOOGLE_APPLICATION_CREDENTIALS']

        self.credentials = GoogleCredentials.from_stream(discovery_file)
        self.max_results = max_results
        self.num_retries = num_retries
        self.service = discovery.build(self.api_name, api_version,
                                       credentials=self.credentials,
                                       discoveryServiceUrl=DISCOVERY_URL)
        self.handle_annotations = handle_annotations
        super(GoogleAPITransformer, self).__init__()

    def _query_api(self, request):
        resource = getattr(self.service, self.resource)()
        request = resource.annotate(body={'requests': request})
        return request.execute(num_retries=self.num_retries)['responses']


class GoogleVisionAPITransformer(GoogleAPITransformer):

    api_name = 'vision'
    resource = 'images'

    def _build_request(self, stims):
        request = []
        for image in stims:
            if image.filename is None:
                file = tempfile.mktemp() + '.png'
                imsave(file, image.data)
            else:
                file = image.filename

            img_data = open(file, 'rb').read()
            content = base64.b64encode(img_data).decode()
            request.append(
                {
                    'image': {'content': content},
                    'features': [{
                        'type': self.request_type,
                        'maxResults': self.max_results,
                    }]
                })

            if image.filename is None:
                os.remove(file)
        return request
