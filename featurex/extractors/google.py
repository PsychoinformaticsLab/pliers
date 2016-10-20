from featurex.extractors.image import ImageExtractor
from featurex.stimuli.image import ImageStim
from featurex.google import GoogleVisionAPITransformer
from featurex import Value, Event
import numpy as np


class GoogleVisionAPIExtractor(GoogleVisionAPITransformer, ImageExtractor):

    def _extract(self, stim):
        if isinstance(stim, ImageStim):
            is_image = True
            stim = [stim]
        else:
            is_image = False
        request =  self._build_request(stim)
        responses = self._query_api(request)

        events = []
        for i, response in enumerate(responses):
            if response:
                annotations = response[self.response_object]
                values = self._parse_annotations(stim[i], annotations)
                onset = stim[i].onset if hasattr(stim[i], 'onset') else i
                ev = Event(onset=onset, duration=stim[i].duration, values=values)
                events.append(ev)
            else:
                events.append(Event())

        if is_image:
            return events[0].values
        return events


class GoogleVisionAPIFaceExtractor(GoogleVisionAPIExtractor):

    request_type = 'FACE_DETECTION'
    response_object = 'faceAnnotations'

    def _parse_annotations(self, stim, annotations):
        values = []
        for annotation in annotations:
            data_dict = {}
            for field, val in annotation.items():
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
            values.append(Value(stim=stim, extractor=self, data=data_dict))
        return values


class GoogleVisionAPILabelExtractor(GoogleVisionAPIExtractor):

    request_type = 'LABEL_DETECTION'
    response_object = 'labelAnnotations'

    def _parse_annotations(self, stim, annotations):
        values = []
        for annotation in annotations:
            data_dict = {field : val for field, val in annotation.items()}
            values.append(Value(stim=stim, extractor=self, data=data_dict))
        return values


class GoogleVisionAPIPropertyExtractor(GoogleVisionAPIExtractor):

    request_type = 'IMAGE_PROPERTIES'
    response_object = 'imagePropertiesAnnotation'

    def _parse_annotations(self, stim, annotation):
        data_dict = {field : val for field, val in annotation.items()}
        return [Value(stim=stim, extractor=self, data=data_dict)]
