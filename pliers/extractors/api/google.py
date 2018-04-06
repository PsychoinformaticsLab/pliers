''' Google API-based feature extraction classes. '''

import base64
from pliers.extractors.image import ImageExtractor
from pliers.extractors.video import VideoExtractor
from pliers.transformers import (GoogleVisionAPITransformer,
                                 GoogleAPITransformer)
from pliers.extractors.base import ExtractorResult
import numpy as np
import pandas as pd
import logging
import time


class GoogleVisionAPIExtractor(GoogleVisionAPITransformer, ImageExtractor):

    ''' Base class for all Extractors that use the Google Vision API. '''

    VERSION = '1.0'

    def _extract(self, stims):
        request = self._build_request(stims)
        responses = self._query_api(request)

        results = []
        for i, response in enumerate(responses):
            if response and self.response_object in response:
                raw = response[self.response_object]
                results.append(ExtractorResult(raw, stims[i], self))
            elif 'error' in response:
                raise Exception(response['error']['message'])
            else:
                results.append(ExtractorResult([{}], stims[i], self))

        return results


class GoogleVisionAPIFaceExtractor(GoogleVisionAPIExtractor):

    ''' Identifies faces in images using the Google Cloud Vision API. '''

    request_type = 'FACE_DETECTION'
    response_object = 'faceAnnotations'

    def _to_df(self, result, handle_annotations=None):
        '''
        Converts a Google API Face JSON response into a Pandas Dataframe.

        Args:
            result (ExtractorResult): Result object from which to parse out a
                Dataframe.
            handle_annotations (str): How returned face annotations should be
                handled in cases where there are multiple faces.
                'first' indicates to only use the first face JSON object, all
                other values will default to including every face.
        '''
        annotations = result._data
        if handle_annotations == 'first':
            annotations = [annotations[0]]

        face_results = []
        for i, annotation in enumerate(annotations):
            data_dict = {}
            for field, val in annotation.items():
                if 'Confidence' in field:
                    data_dict['face_' + field] = val
                elif 'oundingPoly' in field:
                    for j, vertex in enumerate(val['vertices']):
                        for dim in ['x', 'y']:
                            name = '%s_vertex%d_%s' % (field, j+1, dim)
                            val = vertex[dim] if dim in vertex else np.nan
                            data_dict[name] = val
                elif field == 'landmarks':
                    for lm in val:
                        name = 'landmark_' + lm['type'] + '_%s'
                        lm_pos = {name %
                                  k: v for (k, v) in lm['position'].items()}
                        data_dict.update(lm_pos)
                else:
                    data_dict[field] = val

            face_results.append(data_dict)

        return pd.DataFrame(face_results)


class GoogleVisionAPILabelExtractor(GoogleVisionAPIExtractor):

    ''' Labels objects in images using the Google Cloud Vision API. '''

    request_type = 'LABEL_DETECTION'
    response_object = 'labelAnnotations'

    def _to_df(self, result):
        res = {label['description']: label['score'] for label in result._data if label}
        return pd.DataFrame([res])


class GoogleVisionAPIPropertyExtractor(GoogleVisionAPIExtractor):

    ''' Extracts image properties using the Google Cloud Vision API. '''

    request_type = 'IMAGE_PROPERTIES'
    response_object = 'imagePropertiesAnnotation'

    def _to_df(self, result):
        colors = result._data['dominantColors']['colors']
        data_dict = {}
        for color in colors:
            rgb = color['color']
            data_dict[(rgb['red'], rgb['green'], rgb['blue'])] = color['score']
        return pd.DataFrame([data_dict])


class GoogleVisionAPISafeSearchExtractor(GoogleVisionAPIExtractor):

    ''' Extracts safe search detection using the Google Cloud Vision API. '''

    request_type = 'SAFE_SEARCH_DETECTION'
    response_object = 'safeSearchAnnotation'

    def _to_df(self, result):
        return pd.DataFrame([result._data])


class GoogleVisionAPIWebEntitiesExtractor(GoogleVisionAPIExtractor):

    ''' Extracts web entities using the Google Cloud Vision API. '''

    request_type = 'WEB_DETECTION'
    response_object = 'webDetection'

    def _to_df(self, result):
        data_dict = {}
        if 'webEntities' in result._data:
            for entity in result._data['webEntities']:
                if 'description' in entity and 'score' in entity:
                    data_dict[entity['description']] = entity['score']
        return pd.DataFrame([data_dict])


class GoogleVideoIntelligenceAPIExtractor(GoogleAPITransformer, VideoExtractor):

    ''' Extracts object features from videos using the Google Vision Video
    Intelligence API.

    Args:
        features (list): List of features to extract
        config (dict): JSON
        timeout (int): Number of seconds to wait for video intelligence
            operation to finish. Defaults to 90 seconds.
        discovery_file (str): path to discovery file containing Google
            application credentials.
        api_version (str): API version to use.
        max_results (int): Max number of results per page.
        num_retries (int): Number of times to retry query on failure.
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
    '''

    api_name = 'videointelligence'
    _log_attributes = ('discovery_file', 'api_version', 'features', 'config')

    def __init__(self, features=['LABEL_DETECTION', 'SHOT_CHANGE_DETECTION',
                                 'EXPLICIT_CONTENT_DETECTION'],
                 segments=None, config=None, timeout=90, discovery_file=None,
                 api_version='v1', max_results=100, num_retries=3,
                 rate_limit=None):
        self.features = features
        self.segments = segments
        self.config = config
        self.timeout = timeout
        super(GoogleVideoIntelligenceAPIExtractor,
              self).__init__(discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)

    def _query_api(self, request):
        request_obj = self.service.videos().annotate(body=request)
        return request_obj.execute(num_retries=self.num_retries)

    def _query_operations(self, name):
        request_obj = self.service.operations().get(name=name)
        return request_obj.execute(num_retries=self.num_retries)

    def _build_request(self, stim):
        with stim.get_filename() as filename:
            with open(filename, 'rb') as f:
                vid_data = f.read()

        context = self.config if self.config else {}
        if self.segments:
            context['segments'] = self.segments

        request = {
            'inputContent': base64.b64encode(vid_data).decode(),
            'features': self.features,
            'videoContext': context
        }

        return request

    def _extract(self, stim):
        op_request = self._build_request(stim)
        operation = self._query_api(op_request)

        operation_start = time.time()
        response = self._query_operations(operation['name'])
        while 'done' not in response and \
              (time.time() - operation_start) < self.timeout:
            response = self._query_operations(operation['name'])
            time.sleep(1)

        if (time.time() - operation_start) >= self.timeout:
            msg = "The extraction reached the timeout limit of %f, which means "\
                  "the API may not have finished analyzing the video and the "\
                  "results may be empty or incomplete." % self.timeout
            logging.warning(msg)

        return ExtractorResult(response, stim, self)

    def _to_df(self, result):
        response = result._data
        print(response.keys())
        print(response['response'].keys())
        for annotation in response['response']['annotationResults']:
            print(annotation)
            print(annotation.keys())  # 'segmentLabelAnnotations', 'shotLabelAnnotations'
            # ['segmentLabelAnnotations', 'shotLabelAnnotations', 'shotAnnotations', 'explicitAnnotation']

        return response


class GoogleVideoAPILabelDetectionExtractor(GoogleVideoIntelligenceAPIExtractor):

    ''' Extracts image labels using the Google Video Intelligence API '''

    def __init__(self, mode='SHOT_MODE', stationary_camera=False, segments=None,
                 timeout=90, discovery_file=None, api_version='v1',
                 max_results=100, num_retries=3, rate_limit=None):
        config = {
            'labelDetectionConfig': {
                'labelDetectionMode': mode
                'stationaryCamera': stationary_camera
            }
        }
        super(GoogleVideoAPILabelDetectionExtractor,
              self).__init__(features=['LABEL_DETECTION'],
                             config=config,
                             timeout=timeout,
                             discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)


class GoogleVideoAPIShotDetectionExtractor(GoogleVideoIntelligenceAPIExtractor):

    ''' Extracts shot changes using the Google Video Intelligence API '''

    def __init__(self, segments=None, config=None, timeout=90,
                 discovery_file=None, api_version='v1', max_results=100,
                 num_retries=3, rate_limit=None):
        super(GoogleVideoAPIShotDetectionExtractor,
              self).__init__(features=['SHOT_CHANGE_DETECTION'],
                             timeout=timeout,
                             discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)


class GoogleVideoAPIExplicitDetectionExtractor(GoogleVideoIntelligenceAPIExtractor):

    ''' Extracts explicit content using the Google Video Intelligence API '''

    def __init__(self, segments=None, config=None, timeout=90,
                 discovery_file=None, api_version='v1', max_results=100,
                 num_retries=3, rate_limit=None):
        super(GoogleVideoAPIExplicitDetectionExtractor,
              self).__init__(features=['EXPLICIT_CONTENT_DETECTION'],
                             segments=segments,
                             config=config,
                             timeout=timeout,
                             discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)

