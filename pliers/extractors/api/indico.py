'''
Extractors that interact with the Indico API.
'''

import logging
import os
from pliers.extractors.image import ImageExtractor
from pliers.extractors.text import TextExtractor
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.transformers import BatchTransformerMixin
from pliers.transformers.api import APITransformer
from pliers.utils import attempt_to_import, verify_dependencies
import pandas as pd
import numpy as np

indicoio = attempt_to_import('indicoio')


class IndicoAPIExtractor(APITransformer, BatchTransformerMixin, Extractor):

    ''' Base class for all Indico API Extractors

    Args:
        api_key (str): A valid API key for the Indico API. Only needs to be
            passed the first time the extractor is initialized.
        models (list): The names of the Indico models to use.
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
        batch_size (int): Number of stims to send per batched API request.
    '''

    _log_attributes = ('api_key', 'models', 'model_names')
    _input_type = ()
    _batch_size = 20
    _env_keys = 'INDICO_APP_KEY'
    VERSION = '1.0'

    def __init__(self, api_key=None, models=None, rate_limit=None,
                 batch_size=None):
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
        super(IndicoAPIExtractor, self).__init__(rate_limit=rate_limit,
                                                 batch_size=batch_size)

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
                logging.warn(str(e))
                return False
            else:
                # If valid key, a data error (None passed) is expected here
                return True

    def _get_tokens(self, stims):
        return [stim.data for stim in stims if stim.data is not None]

    def _extract(self, stims):
        stims = list(stims)
        tokens = self._get_tokens(stims)
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

    Args:
        api_key (str): A valid API key for the Indico API. Only needs to be
            passed the first time the extractor is initialized.
        models (list): The names of the Indico models to use.
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
        batch_size (int): Number of stims to send per batched API request.
    '''

    def __init__(self, api_key=None, models=None, rate_limit=None,
                 batch_size=None):
        verify_dependencies(['indicoio'])
        self.allowed_models = indicoio.TEXT_APIS.keys()
        super(IndicoAPITextExtractor, self).__init__(api_key=api_key,
                                                     models=models,
                                                     rate_limit=rate_limit,
                                                     batch_size=batch_size)


class IndicoAPIImageExtractor(ImageExtractor, IndicoAPIExtractor):

    ''' Uses to Indico API to extract features from Images, such as
    facial emotion recognition or content filtering.

    Args:
        api_key (str): A valid API key for the Indico API. Only needs to be
            passed the first time the extractor is initialized.
        models (list): The names of the Indico models to use.
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
        batch_size (int): Number of stims to send per batched API request.
    '''

    def __init__(self, api_key=None, models=None, rate_limit=None,
                 batch_size=None):
        verify_dependencies(['indicoio'])
        self.allowed_models = indicoio.IMAGE_APIS.keys()
        super(IndicoAPIImageExtractor, self).__init__(api_key=api_key,
                                                      models=models,
                                                      rate_limit=rate_limit,
                                                      batch_size=batch_size)

    def _get_tokens(self, stims):
        toks = []
        for s in stims:
            if s.url:
                toks.append(s.url)
            elif s.data is not None:
                # IndicoIO breaks if given subclasses of ndarray, and data is
                # an imageio Image instance, so we explicitly convert.
                toks.append(np.array(s.data))
        return toks
