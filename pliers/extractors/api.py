'''
Extractors that interact with external (e.g., deep learning) services.
'''

import os
try:
    from contextlib import ExitStack
except:
    from contextlib2 import ExitStack
from pliers.extractors.image import ImageExtractor
from pliers.extractors.text import TextExtractor
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.transformers import BatchTransformerMixin


try:
    from clarifai.client import ClarifaiApi
except ImportError:
    pass

try:
    import indicoio as ico
except ImportError:
    pass


class IndicoAPIExtractor(BatchTransformerMixin, Extractor):

    ''' Base class for all Indico API Extractors

    Args:
        api_key (str): A valid API key for the Indico API. Only needs to be
            passed the first time the extractor is initialized.
        models (list): The names of the Indico models to use.
    '''

    _log_attributes = ('models',)
    _input_type = ()
    _batch_size = 20

    def __init__(self, api_key=None, models=None):
        super(IndicoAPIExtractor, self).__init__()
        if api_key is None:
            try:
                self.api_key = os.environ['INDICO_APP_KEY']
            except KeyError:
                raise ValueError("A valid Indico API Key "
                                 "must be passed the first time an Indico "
                                 "extractor is initialized.")
        else:
            self.api_key = api_key
        ico.config.api_key = self.api_key

        if models is None:
            raise ValueError("Must enter a valid list of models to use. "
                             "Valid models: {}".format(", ".join(self.allowed_models)))
        for model in models:
            if model not in self.allowed_models:
                raise ValueError(
                "Unsupported model {} specified. "
                "Valid models: {}".format(model, ", ".join(self.allowed_models)))

        self.models = [getattr(ico, model) for model in models]
        self.names = models

    def _extract(self, stims):
        tokens = [stim.data for stim in stims if stim.data is not None]
        scores = [model(tokens) for model in self.models]

        results = []
        for i, stim in enumerate(stims):
            features, data = [], []
            for j, score in enumerate(scores):
                if isinstance(score[i], float):
                    features.append(self.names[j])
                    data.append(score[i])
                elif isinstance(score[i], dict):
                    for k in score[i].keys():
                        features.append(self.names[j] + '_' + k)
                        data.append(score[i][k])

            results.append(ExtractorResult([data], stim, self,
                                           features=features,
                                           onsets=stim.onset,
                                           durations=stim.onset))

        return results


class IndicoAPITextExtractor(TextExtractor, IndicoAPIExtractor):

    ''' Uses to Indico API to extract features from text, such as
    sentiment extraction.
    '''

    def __init__(self, **kwargs):
        self.allowed_models = ico.TEXT_APIS.keys()
        super(IndicoAPITextExtractor, self).__init__(**kwargs)


class IndicoAPIImageExtractor(ImageExtractor, IndicoAPIExtractor):

    ''' Uses to Indico API to extract features from Images, such as
    facial emotion recognition or content filtering.
    '''

    def __init__(self, **kwargs):
        self.allowed_models = ico.IMAGE_APIS.keys()
        super(IndicoAPIImageExtractor, self).__init__(**kwargs)


class ClarifaiAPIExtractor(BatchTransformerMixin, ImageExtractor):

    ''' Uses the Clarifai API to extract tags of images.
    Args:
        app_id (str): A valid APP_ID for the Clarifai API. Only needs to be
            passed the first time the extractor is initialized.
        app_secret (str): A valid APP_SECRET for the Clarifai API. Only needs
            to be passed the first time the extractor is initialized.
        model (str): The name of the Clarifai model to use. If None, defaults
            to the general image tagger.
        select_classes (list): List of classes (strings) to query from the API.
            For example, ['food', 'animal'].
    '''

    _log_attributes = ('model', 'select_classes')
    _batch_size = 128

    def __init__(self, app_id=None, app_secret=None, model=None,
                 select_classes=None):
        ImageExtractor.__init__(self)
        if app_id is None or app_secret is None:
            try:
                app_id = os.environ['CLARIFAI_APP_ID']
                app_secret = os.environ['CLARIFAI_APP_SECRET']
            except KeyError:
                raise ValueError("A valid Clarifai API APP_ID and APP_SECRET "
                                 "must be passed the first time a Clarifai "
                                 "extractor is initialized.")

        self.tagger = ClarifaiApi(app_id=app_id, app_secret=app_secret)
        if model is not None:
            self.tagger.set_model(model)

        self.model = model

        if select_classes is None:
            self.select_classes = None
        else:
            self.select_classes = ','.join(select_classes)

        self._batch_size = self.tagger.get_info()['max_batch_size']

    def _extract(self, stims):
        # Clarifai client expects a list of open file pointers
        # ExitStack lets us use several file context managers simultaneous
        with ExitStack() as stack:
            files = [stack.enter_context(s.get_filename()) for s in stims]
            fps = [stack.enter_context(open(f, 'rb')) for f in files]
            tags = self.tagger.tag_images(fps, select_classes=self.select_classes)

        extracted = []
        for i, res in enumerate(tags['results']):
            tagged = res['result']['tag']
            extracted.append(ExtractorResult([tagged['probs']], stims[i],
                             self, features=tagged['classes']))

        return extracted
