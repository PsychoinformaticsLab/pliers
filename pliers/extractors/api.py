'''
Extractors that interact with external (e.g., deep learning) services.
'''

from pliers.extractors.image import ImageExtractor
from pliers.extractors.text import ComplexTextExtractor
from pliers.extractors.base import ExtractorResult
from scipy.misc import imsave
import os
import tempfile

try:
    from clarifai.client import ClarifaiApi
except ImportError:
    pass

try:
    import indicoio as ico
except ImportError:
    pass

class IndicoAPIExtractor(ComplexTextExtractor):

    ''' Uses the Indico API to extract sentiment of text.
    Args:
        app_key (str): A valid API key for the Indico API. Only needs to be
            passed the first time the extractor is initialized.
        models (list): The names of the Indico models to use.  
    '''

    _log_attributes = ('models',)

    def __init__(self, api_key=None, models=None):
        ComplexTextExtractor.__init__(self)
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
            raise ValueError("Must enter a valid list of models to use of "
                            "possible types: sentiment, sentiment_hq, emotion.")
        else:
            try:
                self.models = [getattr(ico, model) for model in models]
                self.names = models
            except AttributeError:
                raise ValueError("Unsupported model specified. Muse use of of the following:\n"
                                "sentiment, sentiment_hq, emotion, text_tags, language, "
                                "political, keywords, people, places, organizations, "
                                "twitter_engagement, personality, personas, text_features")

    def _extract(self, stim):
        tokens = [token.text for token in stim]
        scores = [model(tokens) for model in self.models]
        data = []
        onsets = []
        durations = []
        for i, w in enumerate(stim):
            features = []
            values = []
            for j, score in enumerate(scores):
                if type(score[i]) == float:
                    features.append(self.names[j])
                    values.append(score[i])
                elif type(score[i]) == dict:
                    for k in score[i].keys():
                        features.append(self.names[j] + '_' + k)
                        values.append(score[i][k])

            data.append(values)
            onsets.append(w.onset)
            durations.append(w.duration)

        return ExtractorResult(data, stim, self, features=features, 
                                onsets=onsets, durations=durations)

class ClarifaiAPIExtractor(ImageExtractor):

    ''' Uses the Clarifai API to extract tags of images.
    Args:
        app_id (str): A valid APP_ID for the Clarifai API. Only needs to be
            passed the first time the extractor is initialized.
        app_secret (str): A valid APP_SECRET for the Clarifai API. 
            Only needs to be passed the first time the extractor is initialized.
        model (str): The name of the Clarifai model to use. 
            If None, defaults to the general image tagger. 
        select_classes (list): List of classes (strings) to query from the API.
            For example, ['food', 'animal'].
    '''

    _log_attributes = ('model', 'select_classes')

    def __init__(self, app_id=None, app_secret=None, model=None, select_classes=None):
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
        if not (model is None):
            self.tagger.set_model(model)

        self.model = model
        
        if select_classes is None:
            self.select_classes = None
        else:
            self.select_classes = ','.join(select_classes)

    def _extract(self, stim):
        if stim.filename is None:
            file = tempfile.mktemp() + '.png'
            imsave(file, stim.data)
        else:
            file = stim.filename
        
        tags = self.tagger.tag_images(open(file, 'rb'), 
                                    select_classes=self.select_classes)
        
        if stim.filename is None:
            os.remove(temp_file)

        tagged = tags['results'][0]['result']['tag']
        return ExtractorResult([tagged['probs']], stim, self, 
                                features=tagged['classes'])
