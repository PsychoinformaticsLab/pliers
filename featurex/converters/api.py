import os
import base64
import json
from abc import abstractmethod
from featurex.stimuli.text import TextStim, ComplexTextStim
from featurex.converters.audio import AudioToTextConverter
from featurex.converters.image import ImageToTextConverter

from six.moves.urllib.parse import urlencode
from six.moves.urllib.request import Request, urlopen
from six.moves.urllib.error import URLError, HTTPError


class SpeechRecognitionConverter(AudioToTextConverter):
    ''' Uses the SpeechRecognition API, which interacts with several APIs, 
    like Google and Wit, to run speech-to-text transcription on an audio file.
    Defaults to using Wit.ai.
    Args:
        api_key (str): API key. Must be passed explicitly or stored in
            the environment variable specified in the environ_key field.
    '''

    def __init__(self, api_key=None):
        import speech_recognition as sr
        if api_key is None:
            try:
                api_key = os.environ[self.environ_key]
            except KeyError:
                raise ValueError("A valid API key must be passed when "
                                 "a SpeechRecognitionConverter is initialized.")
        self.recognizer = sr.Recognizer()
        self.api_key = api_key

    @abstractmethod
    def _convert(self, audio):
        import speech_recognition as sr
        with sr.AudioFile(audio.filename) as source:
            clip = self.recognizer.record(source)
        text = getattr(self.recognizer, self.recognize_method)(clip, self.api_key)
        return ComplexTextStim.from_text(text=text)


class WitTranscriptionConverter(SpeechRecognitionConverter):
    
    environ_key = 'WIT_AI_API_KEY'
    recognize_method = 'recognize_wit'

    def _convert(self, audio):
        return super(WitTranscriptionConverter, self)._convert(audio)


class GoogleSpeechAPIConverter(SpeechRecognitionConverter):

    environ_key = 'GOOGLE_API_KEY'
    recognize_method = 'recognize_google'

    def _convert(self, audio):
        return super(WitTranscriptionConverter, self)._convert(audio)


class IBMSpeechAPIConverter(AudioToTextConverter):
    ''' Uses the IBM Watson Text to Speech API to run speech-to-text 
    transcription on an audio file.

    Args:
        username (str): API credential username. Must be passed explicitly
            or stored in the environment variable specified in the environ_key
            field.
        password (str): API credential password. Must be passed explicitly
            or stored in the environment variable specified in the environ_key
            field.
    '''

    def __init__(self, username=None, password=None):
        import speech_recognition as sr
        if username is None or password is None:
            try:
                username = os.environ['IBM_USERNAME']
                password = os.environ['IBM_PASSWORD']
            except KeyError:
                raise ValueError("A valid API key must be passed when "
                                 "a SpeechRecognitionConverter is initialized.")
        self.recognizer = sr.Recognizer()
        self.username = username
        self.password = password

    def _convert(self, audio):
        import speech_recognition as sr
        with sr.AudioFile(audio.filename) as source:
            clip = self.recognizer.record(source)

        result = self._query_api(clip)

        timestamps = result['results'][0]['alternatives'][0]['timestamps']
        elements = []
        for i, entry in enumerate(timestamps):
            elements.append(DynamicTextStim(text=entry[0], onset=entry[1],
                                                duration=entry[2]-entry[1]))
        
        return ComplexTextStim(elements=elements)


    def _query_api(self, clip):
        # Adapted from SpeechRecognition source code, modified to get text onsets
        flac_data = clip.get_flac_data(
            convert_rate = None if clip.sample_rate >= 16000 else 16000,
            convert_width = None if clip.sample_width >= 2 else 2
        )
        model = "{0}_BroadbandModel".format("en-US")
        url = "https://stream.watsonplatform.net/speech-to-text/api/v1/recognize?{0}".format(urlencode({
            "profanity_filter": "false",
            "continuous": "true",
            "model": model,
            "timestamps": "true",
        }))
        request = Request(url, data = flac_data, headers = {
            "Content-Type": "audio/x-flac",
            "X-Watson-Learning-Opt-Out": "true",
        })
        
        if hasattr("", "encode"): # Python 2.6 compatibility
            authorization_value = base64.standard_b64encode("{0}:{1}".format(self.username, self.password).encode("utf-8")).decode("utf-8")
        else:
            authorization_value = base64.standard_b64encode("{0}:{1}".format(self.username, self.password))
        request.add_header("Authorization", "Basic {0}".format(authorization_value))
        
        try:
            response = urlopen(request, timeout=None)
        except HTTPError as e:
            raise RequestError("recognition request failed: {0}".format(getattr(e, "reason", "status {0}".format(e.code))))
        except URLError as e:
            raise RequestError("recognition connection failed: {0}".format(e.reason))
        
        response_text = response.read().decode("utf-8")
        result = json.loads(response_text)
        return result
