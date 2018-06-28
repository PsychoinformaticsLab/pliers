''' Wit.ai API-based Converters '''

import logging
import os
from abc import abstractproperty
from pliers.stimuli.text import ComplexTextStim
from pliers.utils import attempt_to_import, verify_dependencies
from pliers.converters.audio import AudioToTextConverter
from pliers.transformers.api import APITransformer
from six.moves.urllib.request import Request, urlopen
from six.moves.urllib.error import HTTPError

sr = attempt_to_import('speech_recognition', 'sr')


class SpeechRecognitionAPIConverter(APITransformer, AudioToTextConverter):

    ''' Uses the SpeechRecognition API, which interacts with several APIs,
    like Google and Wit, to run speech-to-text transcription on an audio file.

    Args:
        api_key (str): API key. Must be passed explicitly or stored in
            the environment variable specified in the _env_keys field.
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
    '''

    _log_attributes = ('api_key', 'recognize_method')
    VERSION = '1.0'

    @abstractproperty
    def recognize_method(self):
        pass

    def __init__(self, api_key=None, rate_limit=None):
        verify_dependencies(['sr'])
        if api_key is None:
            try:
                api_key = os.environ[self.env_keys[0]]
            except KeyError:
                raise ValueError("A valid API key must be passed when a"
                                 " SpeechRecognitionAPIConverter is initialized.")
        self.recognizer = sr.Recognizer()
        self.api_key = api_key
        super(SpeechRecognitionAPIConverter, self).__init__(rate_limit=rate_limit)

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

    @property
    def api_keys(self):
        return [self.api_key]

    def check_valid_keys(self):
        url = "https://api.wit.ai/message?v=20160526&q=authenticate"
        request = Request(url, headers={
            "Authorization": "Bearer {}".format(self.api_key)
        })
        try:
            urlopen(request)
            return True
        except HTTPError as e:
            logging.warn(str(e))
            return False
