import os
from featurex.stimuli.text import ComplexTextStim
from featurex.converters.audio import AudioToTextConverter

try:
    import speech_recognition as sr
except ImportError:
    pass


class SpeechRecognitionConverter(AudioToTextConverter):
    ''' Uses the Wit.AI API (via the SpeechRecognition package) to run speech-
    to-text transcription on an audio file.
    Args:
        api_key (str): Wit.AI API key. Must be passed explicitly or stored in
            the environment variable WIT_AI_API_KEY.
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

