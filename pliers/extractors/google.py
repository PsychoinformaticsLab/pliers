''' Google API-based feature extraction classes. '''

from pliers.extractors.image import ImageExtractor
from pliers.stimuli.image import ImageStim
from pliers.google import GoogleVisionAPITransformer
from pliers.extractors.base import ExtractorResult
import numpy as np


class GoogleVisionAPIExtractor(GoogleVisionAPITransformer, ImageExtractor):

    ''' Base class for all Extractors that use the Google Vision API. '''

    def _extract(self, stim):
        if isinstance(stim, ImageStim):
            stim = [stim]

        request = self._build_request(stim)
        responses = self._query_api(request)

        features = []
        data = []
        for i, response in enumerate(responses):
            if response and self.response_object in response:
                annotations = response[self.response_object]
                feat, values = self._parse_annotations(annotations)
                features += feat
                data += values
            elif 'error' in response:
                raise Exception(response['error']['message'])

        data = [data]
        onsets = [stim[i].onset if hasattr(
            stim[i], 'onset') else i for i in range(len(responses))]
        durations = [stim[i].duration for i in range(len(responses))]
        return ExtractorResult(data, stim, self, features=features,
                               onsets=onsets, durations=durations)


class GoogleVisionAPIFaceExtractor(GoogleVisionAPIExtractor):

    ''' Identifies faces in images using the Google Cloud Vision API. '''

    request_type = 'FACE_DETECTION'
    response_object = 'faceAnnotations'

    def _parse_annotations(self, annotations):
        features = []
        values = []

        if self.handle_annotations == 'first':
            annotations = [annotations[0]]

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

            names = list(data_dict.keys())
            if self.handle_annotations == 'prefix' and len(annotations) > 1:
                names = ['face%d_%s' % (i+1, n) for n in names]
            features += names
            values += list(data_dict.values())

        return features, values


class GoogleVisionAPILabelExtractor(GoogleVisionAPIExtractor):

    ''' Labels objects in images using the Google Cloud Vision API. '''

    request_type = 'LABEL_DETECTION'
    response_object = 'labelAnnotations'

    def _parse_annotations(self, annotations):
        features = []
        values = []
        for annotation in annotations:
            features.append(annotation['description'])
            values.append(annotation['score'])
        return features, values


class GoogleVisionAPIPropertyExtractor(GoogleVisionAPIExtractor):

    ''' Extracts image properties using the Google Cloud Vision API. '''

    request_type = 'IMAGE_PROPERTIES'
    response_object = 'imagePropertiesAnnotation'

    def _parse_annotations(self, annotation):
        colors = annotation['dominantColors']['colors']
        features = []
        values = []
        for color in colors:
            rgb = color['color']
            features.append((rgb['red'], rgb['green'], rgb['blue']))
            values.append(color['score'])
        return features, values

class GoogleVisionAPISafeSearchExtractor(GoogleVisionAPIExtractor):

    ''' Extracts safe search detection using the Google Cloud Vision API. '''

    request_type = 'SAFE_SEARCH_DETECTION'
    response_object = 'safeSearchAnnotation'

    def _parse_annotations(self, annotation):
        return annotation.keys(), annotation.values()
