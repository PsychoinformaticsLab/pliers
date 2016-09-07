'''
Extractors that interact with external (e.g., deep learning) services.
'''

from featurex.extractors.image import ImageExtractor
from featurex.extractors.audio import AudioExtractor
from featurex.extractors.text import ComplexTextExtractor
from scipy.misc import imsave
from featurex.core import Value
import os
import tempfile

try:
    from clarifai.client import ClarifaiApi
except ImportError:
    pass

try:
    import speech_recognition as sr
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
            raise ValueError("A valid Indico API Key "                            
                             "must be passed the first time an Indico "
                             "extractor is initialized.")
        else:
            self.api_key = api_key
            ico.config.api_key = self.api_key
        if model is None:
            raise ValueError("Must enter a valid model to use of possible type: "
                             "sentiment, sentiment_hq, emotion.")
        elif model == 'sentiment':
            self.model = ico.sentiment
        elif model == 'sentiment_hq':
            self.model = ico.sentiment_hq
        elif model == 'emotion':
            self.model = ico.emotion
        elif model == 'text_tages':
            self.model = ico.text_tags
        elif model == 'language':
            self.model = ico.language
        elif model == 'political':
            self.model = ico.political
        elif model == 'keywords':
            self.model = ico.keywords
        elif model == 'people':
            self.model = ico.people
        elif model == 'places':
            self.model = ico.places
        elif model == 'organizations':
            self.model = ico.organizations
        elif model == 'twitter_engagement':
            self.model = ico.twitter_engagement
        elif model == 'personality':
            self.model = ico.personality
        elif model == 'personas':
            self.model = ico.personas
        elif model == 'text_features':
            self.model = ico.text_features

    def apply(self, text):
        tokens = [token.text for token in text]
        scores = self.model(tokens)
        return Value(text, self, {'scores': scores})

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

    def apply(self, img):
        data = img.data
        temp_file = tempfile.mktemp() + '.png'
        imsave(temp_file, data)
        tags = self.tagger.tag_images(open(temp_file, 'rb'), select_classes=self.select_classes)
        os.remove(temp_file)

        return Value(img, self, {'tags': tags})


class WitTranscriptionExtractor(AudioExtractor):
    ''' Uses the Wit.AI API (via the SpeechRecognition package) to run speech-
    to-text transcription on an audio file.
    Args:
        api_key (str): Wit.AI API key. Must be passed explicitly or stored in
            the environment variable WIT_AI_API_KEY.
    '''

    def __init__(self, api_key=None):
        if api_key is None:
            try:
                api_key = os.environ['WIT_AI_API_KEY']
            except KeyError:
                raise ValueError("A valid Wit.AI API key must be passed when "
                                 "a WitTranscriptionExtractor is initialized.")
        self.recognizer = sr.Recognizer()
        self.api_key = api_key

    def apply(self, audio):
        with sr.AudioFile(audio.filename) as source:
            clip = self.recognizer.record(source)
        text = self.recognizer.recognize_wit(clip, self.api_key)
        return Value(text, self, {'text': text})
