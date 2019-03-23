'''
Extractors that interact with the Clarifai API.
'''

import logging
import os
try:
    from contextlib import ExitStack
except Exception as e:
    from contextlib2 import ExitStack
from pliers.extractors.image import ImageExtractor
from pliers.extractors.video import VideoExtractor
from pliers.extractors.base import ExtractorResult
from pliers.transformers import BatchTransformerMixin
from pliers.transformers.api import APITransformer
from pliers.utils import listify, attempt_to_import, verify_dependencies
import pandas as pd

clarifai_client = attempt_to_import('clarifai.rest.client', 'clarifai_client',
                                    ['ClarifaiApp',
                                     'Concept',
                                     'ModelOutputConfig',
                                     'ModelOutputInfo',
                                     'Image',
                                     'Video'])


class ClarifaiAPIExtractor(APITransformer):

    ''' Uses the Clarifai API to extract tags of visual stimuli.

    Args:
        api_key (str): A valid API_KEY for the Clarifai API. Only needs to be
            passed the first time the extractor is initialized.
        model (str): The name of the Clarifai model to use. If None, defaults
            to the general image tagger.
        min_value (float): A value between 0.0 and 1.0 indicating the minimum
            confidence required to return a prediction. Defaults to 0.0.
        max_concepts (int): A value between 0 and 200 indicating the maximum
            number of label predictions returned.
        select_concepts (list): List of concepts (strings) to query from the
            API. For example, ['food', 'animal'].
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
    '''

    _log_attributes = ('api_key', 'model', 'model_name', 'min_value',
                       'max_concepts', 'select_concepts')
    _env_keys = ('CLARIFAI_API_KEY',)
    VERSION = '1.0'

    def __init__(self, api_key=None, model='general-v1.3', min_value=None,
                 max_concepts=None, select_concepts=None, rate_limit=None,
                 batch_size=None):
        verify_dependencies(['clarifai_client'])
        if api_key is None:
            try:
                api_key = os.environ['CLARIFAI_API_KEY']
            except KeyError:
                raise ValueError("A valid Clarifai API API_KEY "
                                 "must be passed the first time a Clarifai "
                                 "extractor is initialized.")

        self.api_key = api_key
        try:
            self.api = clarifai_client.ClarifaiApp(api_key=api_key)
            self.model = self.api.models.get(model)
        except clarifai_client.ApiError as e:
            logging.warn(str(e))
            self.api = None
            self.model = None
        self.model_name = model
        self.min_value = min_value
        self.max_concepts = max_concepts
        self.select_concepts = select_concepts
        if select_concepts:
            select_concepts = listify(select_concepts)
            self.select_concepts = [clarifai_client.Concept(concept_name=n)
                                    for n in select_concepts]
        super(ClarifaiAPIExtractor, self).__init__(rate_limit=rate_limit)

    @property
    def api_keys(self):
        return [self.api_key]

    def check_valid_keys(self):
        return self.api is not None

    def _query_api(self, objects):
        verify_dependencies(['clarifai_client'])
        moc = clarifai_client.ModelOutputConfig(min_value=self.min_value,
                                                max_concepts=self.max_concepts,
                                                select_concepts=self.select_concepts)
        model_output_info = clarifai_client.ModelOutputInfo(output_config=moc)
        tags = self.model.predict(objects, model_output_info=model_output_info)
        return tags['outputs']

    def _parse_annotations(self, annotation):
        data_dict = {}
        for tag in annotation['data']['concepts']:
            data_dict[tag['name']] = tag['value']
        return data_dict


class ClarifaiAPIImageExtractor(ClarifaiAPIExtractor, BatchTransformerMixin,
                                ImageExtractor):

    ''' Uses the Clarifai API to extract tags of images.

    Args:
        api_key (str): A valid API_KEY for the Clarifai API. Only needs to be
            passed the first time the extractor is initialized.
        model (str): The name of the Clarifai model to use. If None, defaults
            to the general image tagger.
        min_value (float): A value between 0.0 and 1.0 indicating the minimum
            confidence required to return a prediction. Defaults to 0.0.
        max_concepts (int): A value between 0 and 200 indicating the maximum
            number of label predictions returned.
        select_concepts (list): List of concepts (strings) to query from the
            API. For example, ['food', 'animal'].
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
        batch_size (int): Number of stims to send per batched API request.
    '''

    _batch_size = 32

    def __init__(self, api_key=None, model='general-v1.3', min_value=None,
                 max_concepts=None, select_concepts=None, rate_limit=None,
                 batch_size=None):
        super(ClarifaiAPIImageExtractor,
              self).__init__(api_key=api_key,
                             model=model,
                             min_value=min_value,
                             max_concepts=max_concepts,
                             select_concepts=select_concepts,
                             rate_limit=rate_limit,
                             batch_size=batch_size)

    def _extract(self, stims):
        verify_dependencies(['clarifai_client'])

        # ExitStack lets us use filename context managers simultaneously
        with ExitStack() as stack:
            imgs = []
            for s in stims:
                if s.url:
                    imgs.append(clarifai_client.Image(url=s.url))
                else:
                    f = stack.enter_context(s.get_filename())
                    imgs.append(clarifai_client.Image(filename=f))
            outputs = self._query_api(imgs)

        extractions = []
        for i, resp in enumerate(outputs):
            extractions.append(ExtractorResult(resp, stims[i], self))
        return extractions

    def _to_df(self, result):
        return pd.DataFrame([self._parse_annotations(result._data)])


class ClarifaiAPIVideoExtractor(ClarifaiAPIExtractor, VideoExtractor):

    def _extract(self, stim):
        verify_dependencies(['clarifai_client'])
        with stim.get_filename() as filename:
            vids = [clarifai_client.Video(filename=filename)]
            outputs = self._query_api(vids)
        return ExtractorResult(outputs, stim, self)

    def _to_df(self, result):
        onsets = []
        durations = []
        data = []
        frames = result._data[0]['data']['frames']
        for i, frame_res in enumerate(frames):
            data.append(self._parse_annotations(frame_res))
            onset = frame_res['frame_info']['time'] / 1000.0
            if (i + 1) == len(frames):
                end = result.stim.duration
            else:
                end = frames[i+1]['frame_info']['time'] / 1000.0
            onsets.append(onset)
            # NOTE: As of Clarifai API v2 and client library 2.6.1, the API
            # returns more frames than it should—at least for some videos.
            # E.g., given a 5.5 second clip, it may return 7 frames, with the
            # last beginning at 6000 ms. Since this appears to be a problem on
            # the Clarifai end, and it's not actually clear how they're getting
            # this imaginary frame (I'm guessing it's the very last frame?),
            # we're not going to do anything about it here, except to make sure
            # that durations aren't negative.
            durations.append(max([end - onset, 0]))

        result._onsets = onsets
        result._durations = durations
        df = pd.DataFrame(data)
        result.features = list(df.columns)
        return df
