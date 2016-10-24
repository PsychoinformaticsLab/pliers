from featurex.extractors.image import ImageExtractor
from featurex.stimuli.image import ImageStim
from featurex.google import GoogleVisionAPITransformer
from featurex.extractors import ExtractorResult
import numpy as np


class GoogleVisionAPIExtractor(GoogleVisionAPITransformer, ImageExtractor):

    def _extract(self, stim):
        if isinstance(stim, ImageStim):
            stim = [stim]

        request =  self._build_request(stim)
        responses = self._query_api(request)

        features = []
        data = []
        for i, response in enumerate(responses):
            if response:
                annotations = response[self.response_object]
                feat, values = self._parse_annotations(annotations)
                features += feat
                data += values

        data = [data]
        onsets = [stim[i].onset if hasattr(stim[i], 'onset') else i for i in range(len(responses))]
        durations = [stim[i].duration for i in range(len(responses))]
        return ExtractorResult(data, stim, self, features=features, 
                                onsets=onsets, durations=durations)


class GoogleVisionAPIFaceExtractor(GoogleVisionAPIExtractor):

    request_type = 'FACE_DETECTION'
    response_object = 'faceAnnotations'
    likelihood_dict = {'UNKNOWN': 0.0, 
                        'VERY_UNLIKELY': 0.1, 
                        'UNLIKELY': 0.3, 
                        'POSSIBLE': 0.5, 
                        'LIKELY': 0.7, 
                        'VERY_LIKELY': 0.9}

    def _parse_annotations(self, annotations):
        features = []
        values = []
        for annotation in annotations:
            data_dict = {}
            for field, val in annotation.items():
                if 'Likelihood' in field:
                    data_dict[field] = self.likelihood_dict[val]
                elif 'Confidence' in field:
                    data_dict['face_' + field] = val
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
                else:
                    data_dict[field] = val
            features += data_dict.keys()
            values += data_dict.values()

        return features, values


class GoogleVisionAPILabelExtractor(GoogleVisionAPIExtractor):

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
