''' Google API-based feature extraction classes. '''

import base64
from pliers.extractors.image import ImageExtractor
from pliers.extractors.video import VideoExtractor
from pliers.transformers import (GoogleVisionAPITransformer,
                                 GoogleAPITransformer)
from pliers.extractors.base import ExtractorResult
import numpy as np
import pandas as pd


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

    api_name = 'videointelligence'

    def _query_api(self, request):
        request_obj = self.service.videos() \
            .annotate(body=request)
        return request_obj.execute(num_retries=self.num_retries)

    def _query_operations(self, name):
        request_obj = self.service.operations().get(name=name)
        return request_obj.execute(num_retries=self.num_retries)

    def _build_request(self, stim):
        with stim.get_filename() as filename:
            with open(filename, 'rb') as f:
                vid_data = f.read()

        content = base64.b64encode(vid_data).decode()
        request = {
            'inputContent': content,
            'features': ['LABEL_DETECTION']
        }

        return request

    def _extract(self, stim):
        op_request = self._build_request(stim)
        operation = self._query_api(op_request)

        response = self._query_operations(operation['name'])
        while 'done' not in response:
            response = self._query_operations(operation['name'])

        return ExtractorResult(response, stim, self)

    def _to_df(self, result):
        response = result._data
        print(response.keys())
        print(response['response'].keys())
        for annotation in response['response']['annotationResults']:
            print(annotation)
            print(annotation.keys())  # 'segmentLabelAnnotations', 'shotLabelAnnotations'

        return response
