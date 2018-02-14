'''
Extractors that interact with external (e.g., deep learning) services.
'''

import os
try:
    from contextlib import ExitStack
except Exception as e:
    from contextlib2 import ExitStack
from pliers.extractors.image import ImageExtractor
from pliers.extractors.text import TextExtractor
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.transformers import BatchTransformerMixin
from pliers.utils import (listify, EnvironmentKeyMixin, attempt_to_import,
                          verify_dependencies)
import pandas as pd

clarifai_client = attempt_to_import('clarifai.rest.client', 'clarifai_client',
                                    ['ClarifaiApp',
                                     'Concept',
                                     'ModelOutputConfig',
                                     'ModelOutputInfo',
                                     'Image'])
indicoio = attempt_to_import('indicoio')


class IndicoAPIExtractor(BatchTransformerMixin, Extractor,
                         EnvironmentKeyMixin):

    ''' Base class for all Indico API Extractors

    Args:
        api_key (str): A valid API key for the Indico API. Only needs to be
            passed the first time the extractor is initialized.
        models (list): The names of the Indico models to use.
    '''

    _log_attributes = ('models', 'model_names')
    _input_type = ()
    _batch_size = 20
    _env_keys = 'INDICO_APP_KEY'
    VERSION = '1.0'

    def __init__(self, api_key=None, models=None):
        verify_dependencies(['indicoio'])
        if api_key is None:
            try:
                self.api_key = os.environ['INDICO_APP_KEY']
            except KeyError:
                raise ValueError("A valid Indico API Key "
                                 "must be passed the first time an Indico "
                                 "extractor is initialized.")
        else:
            self.api_key = api_key
        indicoio.config.api_key = self.api_key

        if models is None:
            raise ValueError("Must enter a valid list of models to use. "
                             "Valid models: {}".format(", ".join(self.allowed_models)))
        for model in models:
            if model not in self.allowed_models:
                raise ValueError("Unsupported model {} specified. "
                                 "Valid models: {}".format(model, ", ".join(self.allowed_models)))

        self.model_names = models
        self.models = [getattr(indicoio, model) for model in models]
        self.names = models
        super(IndicoAPIExtractor, self).__init__()

    def _extract(self, stims):
        tokens = [stim.data for stim in stims if stim.data is not None]
        scores = [model(tokens) for model in self.models]

        results = []
        for i, stim in enumerate(stims):
            stim_scores = [s[i] for s in scores]
            results.append(ExtractorResult(None, stim, self,
                                           raw=stim_scores))
        return results

    def _to_df(self, result):
        data_dict = {}
        for i, model_response in enumerate(result.raw):
            if isinstance(model_response, float):
                data_dict[self.names[i]] = model_response
            elif isinstance(model_response, dict):
                for k, v in model_response.items():
                    data_dict['%s_%s' % (self.names[i], k)] = v
        return pd.DataFrame([data_dict])


class IndicoAPITextExtractor(TextExtractor, IndicoAPIExtractor):

    ''' Uses to Indico API to extract features from text, such as
    sentiment extraction.
    '''

    def __init__(self, api_key=None, models=None):
        verify_dependencies(['indicoio'])
        self.allowed_models = indicoio.TEXT_APIS.keys()
        super(IndicoAPITextExtractor, self).__init__(api_key=api_key,
                                                     models=models)


class IndicoAPIImageExtractor(ImageExtractor, IndicoAPIExtractor):

    ''' Uses to Indico API to extract features from Images, such as
    facial emotion recognition or content filtering.
    '''

    def __init__(self, api_key=None, models=None):
        verify_dependencies(['indicoio'])
        self.allowed_models = indicoio.IMAGE_APIS.keys()
        super(IndicoAPIImageExtractor, self).__init__(api_key=api_key,
                                                      models=models)


class ClarifaiAPIExtractor(BatchTransformerMixin, ImageExtractor,
                           EnvironmentKeyMixin):

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
    '''

    _log_attributes = ('model', 'model_name', 'min_value', 'max_concepts',
                       'select_concepts')
    _batch_size = 128
    _env_keys = ('CLARIFAI_API_KEY',)
    VERSION = '1.0'

    def __init__(self, api_key=None, model='general-v1.3', min_value=None,
                 max_concepts=None, select_concepts=None):
        verify_dependencies(['clarifai_client'])
        if api_key is None:
            try:
                api_key = os.environ['CLARIFAI_API_KEY']
            except KeyError:
                raise ValueError("A valid Clarifai API API_KEY "
                                 "must be passed the first time a Clarifai "
                                 "extractor is initialized.")

        self.api = clarifai_client.ClarifaiApp(api_key=api_key)
        self.model_name = model
        self.model = self.api.models.get(model)
        self.min_value = min_value
        self.max_concepts = max_concepts
        self.select_concepts = select_concepts
        if select_concepts:
            select_concepts = listify(select_concepts)
            self.select_concepts = [clarifai_client.Concept(concept_name=n)
                                    for n in select_concepts]
        super(ClarifaiAPIExtractor, self).__init__()

    def _extract(self, stims):
        verify_dependencies(['clarifai_client'])
        moc = clarifai_client.ModelOutputConfig(min_value=self.min_value,
                                                max_concepts=self.max_concepts,
                                                select_concepts=self.select_concepts)
        output_config = moc
        model_output_info = clarifai_client.ModelOutputInfo(output_config=output_config)

        # ExitStack lets us use filename context managers simultaneously
        with ExitStack() as stack:
            files = [stack.enter_context(s.get_filename()) for s in stims]
            imgs = [clarifai_client.Image(filename=filename) for filename in files]
            tags = self.model.predict(imgs, model_output_info=model_output_info)

        extracted = []
        for i, resp in enumerate(tags['outputs']):
            extracted.append(ExtractorResult(None, stims[i], self, raw=resp))
        return extracted

    def _to_df(self, result):
        data_dict = {}
        for tag in result.raw['data']['concepts']:
            data_dict[tag['name']] = tag['value']
        return pd.DataFrame([data_dict])
