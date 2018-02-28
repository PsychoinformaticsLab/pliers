''' Wit.ai API-based Converters '''

import os
from abc import abstractproperty
from pliers.stimuli.text import ComplexTextStim
from pliers.utils import attempt_to_import, verify_dependencies
from pliers.converters.audio import AudioToTextConverter
from pliers.transformers.api import APITransformer

sr = attempt_to_import('speech_recognition', 'sr')


class SpeechRecognitionAPIConverter(APITransformer, AudioToTextConverter):

    ''' Uses the SpeechRecognition API, which interacts with several APIs,
    like Google and Wit, to run speech-to-text transcription on an audio file.

    Args:
        api_key (str): API key. Must be passed explicitly or stored in
            the environment variable specified in the _env_keys field.
    '''

    _log_attributes = ('recognize_method',)
    VERSION = '1.0'

    @abstractproperty
    def recognize_method(self):
        pass

    def __init__(self, api_key=None):
        verify_dependencies(['sr'])
        if api_key is None:
            try:
                api_key = os.environ[self.env_keys[0]]
            except KeyError:
                raise ValueError("A valid API key must be passed when a"
                                 " SpeechRecognitionAPIConverter is initialized.")
        self.recognizer = sr.Recognizer()
        self.api_key = api_key
        super(SpeechRecognitionAPIConverter, self).__init__()

    def _convert(self, audio):
        verify_dependencies(['sr'])
        with audio.get_filename() as filename:
            with sr.AudioFile(filename) as source:
                clip = self.recognizer.record(source)

        text = getattr(self.recognizer, self.recognize_method)(clip, self.api_key)

        return ComplexTextStim(text=text)


class WitTranscriptionConverter(SpeechRecognitionAPIConverter):

    ''' Speech-to-text transcription via the Wit.ai API. '''

    _env_keys = 'WIT_AI_API_KEY'
    recognize_method = 'recognize_wit'
