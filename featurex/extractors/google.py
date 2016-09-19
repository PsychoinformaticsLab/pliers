# export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials-key.json

import argparse
import base64
import os
import re
import sys
from featurex.extractors.image import ImageExtractor
from featurex.stimuli.image import ImageStim
from featurex import Value, Event
import tempfile
from scipy.misc import imsave
import numpy as np

try:
    from googleapiclient import discovery, errors
    from oauth2client.client import GoogleCredentials
except ImportError:
    pass


DISCOVERY_URL = 'https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'
BATCH_SIZE = 10


class GoogleAPIExtractor(ImageExtractor):

    def __init__(self, discovery_file=None, api_version='v1', max_results=100,
                 num_retries=3):

        if discovery_file is None:
            if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
                raise ValueError("No Google application credentials found. "
                    "A JSON service account key must be either passed as the "
                    "discovery_file argument, or set in the "
                    "GOOGLE_APPLICATION_CREDENTIALS environment variable.")
            discovery_file = os.environ['GOOGLE_APPLICATION_CREDENTIALS']

        self.credentials = GoogleCredentials.from_stream(discovery_file)
        self.max_results = max_results
        self.num_retries = num_retries
        self.service = discovery.build(self.api_name, api_version,
                                       credentials=self.credentials,
                                       discoveryServiceUrl=DISCOVERY_URL)
        super(GoogleAPIExtractor, self).__init__()

    def apply(self, stim):
        if isinstance(stim, ImageStim):
            is_image = True
            stim = [stim]
        else:
            is_image = False
        request =  self._build_request(stim)
        responses = self._query_api(request)

        events = []
        for i, response in enumerate(responses):
            #TODO: what if response is empty
            #TODO: don't only use the first annotation
            annotations = response[self.response_object]
            value = self._parse_annotations(stim[i], annotations)
            onset = stim[i].onset if hasattr(stim[i], 'onset') else i
            ev = Event(onset=onset, duration=stim[i].duration, values=[value])
            events.append(ev)

        if is_image:
            return events[0].values[0]
        return events

    def _query_api(self, request):
        resource = getattr(self.service, self.resource)()
        request = resource.annotate(body={'requests': request})
        return request.execute(num_retries=self.num_retries)['responses']


class GoogleVisionAPIExtractor(GoogleAPIExtractor):

    api_name = 'vision'
    resource = 'images'

    def _build_request(self, stim):
        request = []
        for image in stim:
            temp_file = tempfile.mktemp() + '.png'
            imsave(temp_file, image.data)
            img_data = open(temp_file, 'rb').read()
            content = base64.b64encode(img_data).decode()
            request.append(
                {
                'image': { 'content': content },
                'features': [{
                    'type': self.request_type,
                    'maxResults': self.max_results,
                }]
            })
        return request


class GoogleVisionAPIFaceExtractor(GoogleVisionAPIExtractor):

    request_type = 'FACE_DETECTION'
    response_object = 'faceAnnotations'

    def _parse_annotations(self, stim, annotations):
        annotations = annotations[0]
        data_dict = {}
        for field, val in annotations.items():
            if field not in ['boundingPoly', 'fdBoundingPoly', 'landmarks']:
                data_dict[field] = val
            elif 'oundingPoly' in field:
                for i, vertex in enumerate(val['vertices']):
                    for dim in ['x', 'y']:
                        name = '%s_vertex%d_%s' % (field, i+1, dim)
                        val = vertex[dim] if dim in vertex else np.nan
                        data_dict[name] = val
            elif field == 'landmarks':
                for lm in val:
                    name = 'landmark_' + lm['type'] + '_%s'
                    lm_pos = { name % k : v for (k, v) in lm['position'].items()}
                    data_dict.update(lm_pos)
        return Value(stim=stim, extractor=self, data=data_dict)


class GoogleVisionAPITextExtractor(GoogleVisionAPIExtractor):

    request_type = 'TEXT_DETECTION'
    response_object = 'textAnnotations'

    def _parse_annotations(self, stim, annotations):
        annotations = annotations[0]
        data_dict = {}
        for field, val in annotations.items():
            if 'boundingPoly' != field:
                data_dict[field] = val
            else:
                for i, vertex in enumerate(val['vertices']):
                    for dim in ['x', 'y']:
                        name = '%s_vertex%d_%s' % (field, i+1, dim)
                        val = vertex[dim] if dim in vertex else np.nan
                        data_dict[name] = val
        return Value(stim=stim, extractor=self, data=data_dict)


class GoogleVisionAPILabelExtractor(GoogleVisionAPIExtractor):

    request_type = 'LABEL_DETECTION'
    response_object = 'labelAnnotations'

    def _parse_annotations(self, stim, annotations):
        annotations = annotations[0]
        data_dict = {field : val for field, val in annotations.items()}
        return Value(stim=stim, extractor=self, data=data_dict)


class GoogleVisionAPIPropertyExtractor(GoogleVisionAPIExtractor):

    request_type = 'IMAGE_PROPERTIES'
    response_object = 'imagePropertiesAnnotation'

    def _parse_annotations(self, stim, annotation):
        data_dict = {field : val for field, val in annotation.items()}
        return Value(stim=stim, extractor=self, data=data_dict)
