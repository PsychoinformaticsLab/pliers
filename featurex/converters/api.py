import os
from featurex.stimuli.audio import AudioStim
from featurex.stimuli.text import TextStim
from featurex.converters.audio import AudioToTextConverter

try:
    import speech_recognition as sr
except ImportError:
    pass


class WitTranscriptionConverter(AudioToTextConverter):
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
                                 "a WitTranscriptionConverter is initialized.")
        self.recognizer = sr.Recognizer()
        self.api_key = api_key

    def _convert(self, audio):
        with sr.AudioFile(audio.filename) as source:
            clip = self.recognizer.record(source)
        text = self.recognizer.recognize_wit(clip, self.api_key)
        return TextStim(text=text)


class GoogleSpeechAPIConverter(AudioToTextConverter):
    ''' 
    Uses the Google Speech API (via the SpeechRecognition package) to run speech-
    to-text transcription on an audio file.
    '''

    def __init__(self, api_key=None):
        if api_key is None:
            try:
                api_key = os.environ['GOOGLE_API_KEY']
            except KeyError:
                raise ValueError("A valid Google API key must be passed when "
                                 "a GoogleSpeechAPIConverter is initialized.")
        self.recognizer = sr.Recognizer()
        self.api_key = api_key

    def _convert(self, audio):
        with sr.AudioFile(audio.filename) as source:
            clip = self.recognizer.record(source)
        text = self.recognizer.recognize_google(clip, key=self.api_key)
        return TextStim(text=text)

