import os
from featurex.stimuli.text import TextStim, ComplexTextStim
from featurex.converters.audio import AudioToTextConverter
from featurex.converters.image import ImageToTextConverter
from PIL import Image

try:
    import speech_recognition as sr
except ImportError:
    pass


class SpeechRecognitionConverter(AudioToTextConverter):
    ''' Uses the SpeechRecognition API, which interacts with several APIs, 
    like Google and Wit, to run speech-to-text transcription on an audio file.
    Args:
        api_key (str): API key. Must be passed explicitly or stored in
            the environment variable specified in the environ_key field.
    '''

    def __init__(self, api_key=None):
        if api_key is None:
            try:
                api_key = os.environ[self.environ_key]
            except KeyError:
                raise ValueError("A valid API key must be passed when "
                                 "a SpeechRecognitionConverter is initialized.")
        self.recognizer = sr.Recognizer()
        self.api_key = api_key

    def _convert(self, audio):
        with sr.AudioFile(audio.filename) as source:
            clip = self.recognizer.record(source)
        text = getattr(self.recognizer, self.recognize_method)(clip, self.api_key)
        return ComplexTextStim.from_text(text=text)


class WitTranscriptionConverter(SpeechRecognitionConverter):
    
    environ_key = 'WIT_AI_API_KEY'
    recognize_method = 'recognize_wit'


class GoogleSpeechAPIConverter(SpeechRecognitionConverter):

    environ_key = 'GOOGLE_API_KEY'
    recognize_method = 'recognize_google'


class TesseractAPIConverter(ImageToTextConverter):
    ''' Uses the Tesseract library to extract text from images '''

    def __init__(self):
        super(self.__class__, self).__init__()

    def _convert(self, image):
        import pytesseract
        text = pytesseract.image_to_string(Image.fromarray(image.data))
        return TextStim(text=text)
