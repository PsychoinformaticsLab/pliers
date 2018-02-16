''' Google API-based feature extraction classes. '''

from pliers.extractors.image import ImageExtractor
from pliers.transformers import GoogleVisionAPITransformer
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
                results.append(ExtractorResult(None, stims[i], self, raw=raw))
            elif 'error' in response:
                raise Exception(response['error']['message'])
            else:
                results.append(ExtractorResult(None, stims[i], self, raw=[{}]))

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
        annotations = result.data
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
        res = {label['description']: label['score'] for label in result.data}
        return pd.DataFrame([res])


class GoogleVisionAPIPropertyExtractor(GoogleVisionAPIExtractor):

    ''' Extracts image properties using the Google Cloud Vision API. '''

    request_type = 'IMAGE_PROPERTIES'
    response_object = 'imagePropertiesAnnotation'

    def _to_df(self, result):
        colors = result.data['dominantColors']['colors']
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
        return pd.DataFrame([result.data])


class GoogleVisionAPIWebEntitiesExtractor(GoogleVisionAPIExtractor):

    ''' Extracts web entities using the Google Cloud Vision API. '''

    request_type = 'WEB_DETECTION'
    response_object = 'webDetection'

    def _to_df(self, result):
        data_dict = {}
        if 'webEntities' in result.data:
            for entity in result.data['webEntities']:
                if 'description' in entity and 'score' in entity:
                    data_dict[entity['description']] = entity['score']
        return pd.DataFrame([data_dict])
