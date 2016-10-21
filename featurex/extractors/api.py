'''
Extractors that interact with external (e.g., deep learning) services.
'''

from featurex.extractors.image import ImageExtractor
from featurex.extractors.audio import AudioExtractor
from featurex.extractors.text import ComplexTextExtractor
from featurex.extractors import ExtractorResult
from scipy.misc import imsave
from featurex.core import Value, Event
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
        model (str): The name of the Indico model to use.  
    '''

    def __init__(self, api_key=None, model=None):
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
        if model is None:
            raise ValueError("Must enter a valid model to use of possible type: "
                             "sentiment, sentiment_hq, emotion.")
        else:
            try:
                self.model = getattr(ico, model)
                self.name = model
            except AttributeError:
                raise ValueError("Unsupported model specified. Muse use of of the following:\n"
                                "sentiment, sentiment_hq, emotion, text_tags, language, "
                                "political, keywords, people, places, organizations, "
                                "twitter_engagement, personality, personas, text_features")

    def _extract(self, text):
        tokens = [token.text for token in text]
        scores = self.model(tokens)
        data = []
        onsets = []
        durations = []
        for i, w in enumerate(text):
            if type(scores[i]) == float:
                features = [self.name]
                values = [scores[i]]
            elif type(scores[i]) == dict:
                features = []
                values = []
                for k in scores[i].keys():
                    features.append(self.name + '_' + k)
                    values.append(scores[i][k])

            data.append(values)
            onsets.append(w.onset)
            durations.append(w.duration)

        return ExtractorResult(data, text, self, features=features, 
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
    '''

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

        self.select_classes = select_classes

    def _extract(self, img):
        data = img.data
        temp_file = tempfile.mktemp() + '.png'
        imsave(temp_file, data)
        tags = self.tagger.tag_images(open(temp_file, 'rb'), select_classes=self.select_classes)
        os.remove(temp_file)

        tagged = tags['results'][0]['result']['tag']
        return ExtractorResult([tagged['probs']], img, self, 
                                features=tagged['classes'])
