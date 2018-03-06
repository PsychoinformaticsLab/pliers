'''
Extractors that interact with the Indico API.
'''

import os
from pliers.extractors.image import ImageExtractor
from pliers.extractors.text import TextExtractor
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.transformers import BatchTransformerMixin
from pliers.transformers.api import APITransformer
from pliers.utils import attempt_to_import, verify_dependencies
import pandas as pd

indicoio = attempt_to_import('indicoio')


class IndicoAPIExtractor(APITransformer, BatchTransformerMixin, Extractor):

    ''' Base class for all Indico API Extractors

    Args:
        api_key (str): A valid API key for the Indico API. Only needs to be
            passed the first time the extractor is initialized.
        models (list): The names of the Indico models to use.
    '''

    _log_attributes = ('api_key', 'models', 'model_names')
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

    @property
    def api_keys(self):
        return [self.api_key]

    def check_valid_keys(self):
        verify_dependencies(['indicoio'])
        from indicoio.utils import api
        from indicoio.utils.errors import IndicoError
        try:
            api.api_handler(None, None, self.model_names[0])
        except IndicoError as e:
            if str(e) == 'Invalid API key':
                return False
            else:
                # If valid key, a data error (None passed) is expected here
                return True

    def _extract(self, stims):
        tokens = [stim.data for stim in stims if stim.data is not None]
        scores = [model(tokens) for model in self.models]

        results = []
        for i, stim in enumerate(stims):
            stim_scores = [s[i] for s in scores]
            results.append(ExtractorResult(stim_scores, stim, self))
        return results

    def _to_df(self, result):
        data_dict = {}
        for i, model_response in enumerate(result._data):
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
