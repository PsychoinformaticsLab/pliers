''' Google API-based feature extraction classes. '''

from pliers.extractors.image import ImageExtractor
from pliers.extractors.text import TextExtractor
from pliers.extractors.video import VideoExtractor
from pliers.transformers import (GoogleAPITransformer,
                                 GoogleVisionAPITransformer,
                                 GoogleAPITransformer)
from pliers.extractors.base import ExtractorResult
from pliers.utils import flatten_dict
import numpy as np
import pandas as pd
import logging
import time
import warnings
import os
from collections import defaultdict


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
            key = [rgb.get('red', 0), rgb.get('green', 0), rgb.get('blue', 0)]
            key = ', '.join([str(v) for v in key])
            data_dict[key] = color['score']
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
        features (list): List of features to extract. LABEL_DETECTION extracts
            tags present throughout the provided segments (full video if none
            provided) as well as throughout the shots (depending on config).
            SHOT_CHANGE_DETECTION extracts a shot feature with onsets and
            durations corresponding to shot changes in the video.
            EXPLICIT_CONTENT_DETECTION extracts any frame onsets of explicit
            material.
        segments (list): List of JSON objects or dictionaries. Each dictionary
            should contain a startTimeOffset and an endTimeOffset field with
            timestamps of the format XX.XXs marking the desired segments upon
            which to extract features.
        config (dict): JSON object representing the desired configuration for
            extraction. See the Google Cloud Video Intelligence documentation
            for more details.
        timeout (int): Number of seconds to wait for video intelligence
            operation to finish. Defaults to 90 seconds.
        request_rate (int): Number of seconds to wait between polling the
            extraction operation for completion.
        discovery_file (str): path to discovery file containing Google
            application credentials.
        api_version (str): API version to use.
        max_results (int): Max number of results per page.
        num_retries (int): Number of times to retry query on failure.
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
    '''

    api_name = 'videointelligence'
    _log_attributes = ('discovery_file', 'api_version', 'features', 'segments',
                       'config', 'timeout', 'request_rate')

    def __init__(self, features=['LABEL_DETECTION', 'SHOT_CHANGE_DETECTION',
                                 'EXPLICIT_CONTENT_DETECTION'],
                 segments=None, config=None, timeout=90, request_rate=5,
                 discovery_file=None, api_version='v1', max_results=100,
                 num_retries=3, rate_limit=None):
        self.features = features
        self.segments = segments
        self.config = config
        self.timeout = timeout
        self.request_rate = request_rate
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

        context = self.config if self.config else {}
        if self.segments:
            context['segments'] = self.segments

        with stim.get_filename() as filename:
            size = os.path.getsize(filename)
            LIMIT = 524288000
            if size > LIMIT:
                warnings.warn("Video file is very large ({} bytes) and may "
                              "exceed the Google Video Intelligence payload "
                              "limit ({} bytes).".format(size, LIMIT))

        request = {
            'inputContent': stim.get_bytestring(),
            'features': self.features,
            'videoContext': context
        }

        return request

    def _extract(self, stim):
        op_request = self._build_request(stim)
        operation = self._query_api(op_request)

        msg = "Beginning video extraction with a timeout of %fs. Even for "\
              "small videos, full extraction may take awhile." % self.timeout
        logging.warning(msg)

        operation_start = time.time()
        response = self._query_operations(operation['name'])
        while 'done' not in response and \
              (time.time() - operation_start) < self.timeout:
            response = self._query_operations(operation['name'])
            time.sleep(self.request_rate)

        if (time.time() - operation_start) >= self.timeout:
            msg = "The extraction reached the timeout limit of %fs, which "\
                  "means the API may not have finished analyzing the video "\
                  "and the results may be empty or incomplete." % self.timeout
            logging.warning(msg)

        return ExtractorResult(response, stim, self)

    def _get_onset_duration(self, timing_json):
        onset = float(timing_json['startTimeOffset'][:-1])
        end = float(timing_json['endTimeOffset'][:-1])
        return onset, (end - onset)

    def _parse_label(self, data, features, label):
        for segment in label.get('segments', []):
            onset, duration = self._get_onset_duration(segment['segment'])
            score = segment['confidence']
            data[(onset, duration)].update({f: score for f in features})

    def _parse_frame(self, data, features, annotation, score_key, max_time):
        frames = annotation.get('frames', [])
        for i, frame in enumerate(frames):
            onset = float(frame['timeOffset'][:-1])
            if (i + 1) == len(frames):
                end = max_time
            else:
                end = float(frames[i+1]['timeOffset'][:-1])
            duration = end - onset
            score = frame[score_key]
            data[(onset, duration)].update({f: score for f in features})

    def _to_df(self, result):
        response = result._data.get('response', {})
        data = defaultdict(dict)
        for r in response.get('annotationResults', []):
            for key, res in r.items():
                if 'Label' in key:
                    for annot in res:
                        feats = [annot['entity']['description']]
                        for category in annot.get('categoryEntities', []):
                            feats.append('category_' + category['description'])
                        if key == 'frameLabelAnnotations':
                            self._parse_frame(data, feats, annot, 'confidence',
                                              result.stim.duration)
                        else:
                            # Good for shot or segment labels
                            self._parse_label(data, feats, annot)
                elif key == 'shotAnnotations':
                    for i, shot in enumerate(res):
                        onset, duration = self._get_onset_duration(shot)
                        data[(onset, duration)].update({
                            'shot_id': i
                        })
                elif key == 'explicitAnnotation':
                    feature = 'pornographyLikelihood'
                    self._parse_frame(data, [feature], res, feature,
                                      result.stim.duration)

        df = pd.DataFrame(list(data.values()))
        # If multiple confidences were parsed, uses the last one
        if len(data) > 0:
            onsets, durations = zip(*list(data.keys()))
            result._onsets = onsets
            result._durations = durations
        result.features = list(df.columns)
        return df


class GoogleVideoAPILabelDetectionExtractor(GoogleVideoIntelligenceAPIExtractor):

    ''' Extracts image labels using the Google Video Intelligence API '''

    def __init__(self, mode='SHOT_MODE', stationary_camera=False,
                 segments=None, timeout=90, request_rate=5, num_retries=3,
                 discovery_file=None, api_version='v1', max_results=100,
                 rate_limit=None, frame_confidence_threshold=None,
                 video_confidence_threshold=None):

        config = {
            'labelDetectionConfig': {
                'labelDetectionMode': mode,
                'stationaryCamera': stationary_camera
            }
        }

        if frame_confidence_threshold is not None:
            if mode not in ['FRAME_MODE', 'SHOT_AND_FRAME_MODE']:
                raise ValueError(
                    "frame_confidence_threshold can only be specified in"
                    "FRAME or SHOT_AND_FRAME modes.")
            else:
                config['labelDetectionConfig']['frameConfidenceThreshold'] = \
                    frame_confidence_threshold

        if video_confidence_threshold is not None:
            if mode not in ['SHOT_MODE', 'SHOT_AND_FRAME_MODE']:
                raise ValueError(
                    "video_confidence_threshold can only be specified in"
                    "SHOT or SHOT_AND_FRAME modes.")
            else:
                config['labelDetectionConfig']['videoConfidenceThreshold'] = \
                    video_confidence_threshold

        super(GoogleVideoAPILabelDetectionExtractor,
              self).__init__(features=['LABEL_DETECTION'],
                             segments=segments,
                             config=config,
                             timeout=timeout,
                             request_rate=request_rate,
                             discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)


class GoogleVideoAPIShotDetectionExtractor(GoogleVideoIntelligenceAPIExtractor):

    ''' Extracts shot changes using the Google Video Intelligence API '''

    def __init__(self, segments=None, config=None, timeout=90, request_rate=5,
                 discovery_file=None, api_version='v1', max_results=100,
                 num_retries=3, rate_limit=None):
        super(GoogleVideoAPIShotDetectionExtractor,
              self).__init__(features=['SHOT_CHANGE_DETECTION'],
                             segments=segments,
                             config=config,
                             timeout=timeout,
                             request_rate=request_rate,
                             discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)


class GoogleVideoAPIExplicitDetectionExtractor(GoogleVideoIntelligenceAPIExtractor):

    ''' Extracts explicit content using the Google Video Intelligence API '''

    def __init__(self, segments=None, config=None, timeout=90, request_rate=5,
                 discovery_file=None, api_version='v1', max_results=100,
                 num_retries=3, rate_limit=None):
        super(GoogleVideoAPIExplicitDetectionExtractor,
              self).__init__(features=['EXPLICIT_CONTENT_DETECTION'],
                             segments=segments,
                             config=config,
                             timeout=timeout,
                             request_rate=request_rate,
                             discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)


class GoogleLanguageAPIExtractor(GoogleAPITransformer, TextExtractor):

    ''' Extracts natural language features from text documents using the
    Google Natural Language API.

    Args:
        features (list): List of features (str) to extract. Available
            features: extractSyntax, extractEntities, extractDocumentSentiment,
            extractEntitySentiment, and classifyText. See Google Natural
            Language API documentation for more details.
        language (str): The ISO-639-1 or BCP-47 identifier for the document
            language. If None is provided, API auto-detects the language.
        is_html (bool): When True, the document's text is expected to be
            HTML. Otherwise, plain text is assumed.
        discovery_file (str): path to discovery file containing Google
            application credentials.
        api_version (str): API version to use.
        max_results (int): Max number of results per page.
        num_retries (int): Number of times to retry query on failure.
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
    '''

    api_name = 'language'
    _log_attributes = ('discovery_file', 'api_version', 'features',
                       'language', 'is_html')

    def __init__(self, features=['extractSyntax',
                                 'extractEntities',
                                 'extractDocumentSentiment',
                                 'extractEntitySentiment',
                                 'classifyText'],
                 language=None, is_html=False, discovery_file=None,
                 api_version='v1', max_results=100,
                 num_retries=3, rate_limit=None):
        self.features = features
        self.language = language
        self.is_html = is_html
        super(GoogleLanguageAPIExtractor,
              self).__init__(discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)

    def _query_api(self, request):
        request_obj = self.service.documents().annotateText(body=request)
        return request_obj.execute(num_retries=self.num_retries)

    def _build_request(self, stim):
        document = {
            'type' : 'HTML' if self.is_html else 'PLAIN_TEXT',
            'content' : stim.text
        }

        if self.language:
            document['language'] = self.language

        request = {
            'document': document,
            'features': { f : True for f in self.features },
            'encodingType': 'UTF32'
        }

        return request

    def _extract(self, stim):
        request = self._build_request(stim)
        response = self._query_api(request)
        return ExtractorResult(response, stim, self)

    def _get_span(self, text_json):
        offset = text_json['text']['beginOffset']
        content = text_json['text']['content']
        return { 'begin_char_index' : offset,
                 'end_char_index' : offset + len(content),
                 'text' : content }

    def _to_df(self, result):
        response = result._data
        data = []

        # One row/object for all document-level features
        document_data = {}

        if 'extractDocumentSentiment' in self.features:
            sentiment = response['documentSentiment']
            document_data.update(flatten_dict(sentiment, 'sentiment'))

            # Sentence level sentiment
            for sentence in response.get('sentences', []):
                sentence_data = self._get_span(sentence)
                sentiment = sentence['sentiment']
                sentence_data.update(flatten_dict(sentiment, 'sentiment'))
                data.append(sentence_data)

        for category in response.get('categories'):
            key = 'category_%s' % category['name']
            document_data[key] = category['confidence']

        # Include only if there are document-level features
        if document_data:
            data.append(document_data)

        # Entity-level features
        for entity in response.get('entities', []):
            entity_copy = entity.copy()
            mentions = entity_copy.pop('mentions', [])
            entity_copy.pop('name', None)
            entity_copy = flatten_dict(entity_copy)

            for m in mentions:
                entity_data = self._get_span(m)
                entity_data.update(entity_copy)
                # Overwrite top-level sentiment with mention-level
                sentiment = m.get('sentiment', {})
                entity_data.update(flatten_dict(sentiment, 'sentiment'))
                data.append(entity_data)

        # Token-level syntax features
        for token in response.get('tokens', []):
            token_data = self._get_span(token)
            token_data['lemma'] = token['lemma']
            token_data.update(token['partOfSpeech'])
            dependency = flatten_dict(token['dependencyEdge'], 'dependency')
            token_data.update(dependency)
            data.append(token_data)

        df = pd.DataFrame(data)
        df['language'] = response['language']
        return df


class GoogleLanguageAPIEntityExtractor(GoogleLanguageAPIExtractor):

    ''' Extracts entity labels in text using the Google Language API '''

    def __init__(self, language=None, is_html=False, discovery_file=None,
                 api_version='v1', max_results=100, num_retries=3,
                 rate_limit=None):
        super(GoogleLanguageAPIEntityExtractor,
              self).__init__(features=['extractEntities'],
                             language=language,
                             is_html=is_html,
                             discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)


class GoogleLanguageAPISentimentExtractor(GoogleLanguageAPIExtractor):

    ''' Extracts sentiment of text using the Google Language API '''

    def __init__(self, language=None, is_html=False, discovery_file=None,
                 api_version='v1', max_results=100, num_retries=3,
                 rate_limit=None):
        super(GoogleLanguageAPISentimentExtractor,
              self).__init__(features=['extractDocumentSentiment'],
                             language=language,
                             is_html=is_html,
                             discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)


class GoogleLanguageAPISyntaxExtractor(GoogleLanguageAPIExtractor):

    ''' Extracts syntax properties of text using the Google Language API '''

    def __init__(self, language=None, is_html=False, discovery_file=None,
                 api_version='v1', max_results=100, num_retries=3,
                 rate_limit=None):
        super(GoogleLanguageAPISyntaxExtractor,
              self).__init__(features=['extractSyntax'],
                             language=language,
                             is_html=is_html,
                             discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)


class GoogleLanguageAPITextCategoryExtractor(GoogleLanguageAPIExtractor):

    ''' Extracts document category using the Google Language API.
    See the API documentation for the taxonomy of categories:
    https://cloud.google.com/natural-language/docs/categories '''

    def __init__(self, language=None, is_html=False, discovery_file=None,
                 api_version='v1', max_results=100, num_retries=3,
                 rate_limit=None):
        super(GoogleLanguageAPITextCategoryExtractor,
              self).__init__(features=['classifyText'],
                             language=language,
                             is_html=is_html,
                             discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)


class GoogleLanguageAPIEntitySentimentExtractor(GoogleLanguageAPIExtractor):

    ''' Extracts sentiment of entities found in text using the Google Language
    API. Produces identical results to the entity extractor but with additional
    sentiment analysis. '''

    def __init__(self, language=None, is_html=False, discovery_file=None,
                 api_version='v1', max_results=100, num_retries=3,
                 rate_limit=None):
        super(GoogleLanguageAPIEntitySentimentExtractor,
              self).__init__(features=['extractEntitySentiment'],
                             language=language,
                             is_html=is_html,
                             discovery_file=discovery_file,
                             api_version=api_version,
                             max_results=max_results,
                             num_retries=num_retries,
                             rate_limit=rate_limit)
