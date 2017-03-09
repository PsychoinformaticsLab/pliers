''' Converters that query external APIs. '''

import os
import base64
import json
import tempfile
from abc import abstractproperty
from pliers.stimuli.text import TextStim, ComplexTextStim
from pliers.transformers import EnvironmentKeyMixin
from .audio import AudioToTextConverter
from six.moves.urllib.parse import urlencode
from six.moves.urllib.request import Request, urlopen
from six.moves.urllib.error import URLError, HTTPError


class SpeechRecognitionAPIConverter(AudioToTextConverter, EnvironmentKeyMixin):

    ''' Uses the SpeechRecognition API, which interacts with several APIs,
    like Google and Wit, to run speech-to-text transcription on an audio file.
    Args:
        api_key (str): API key. Must be passed explicitly or stored in
            the environment variable specified in the _env_keys field.
    '''

    @abstractproperty
    def recognize_method(self):
        pass

    def __init__(self, api_key=None):
        super(SpeechRecognitionAPIConverter, self).__init__()
        import speech_recognition as sr
        if api_key is None:
            try:
                api_key = os.environ[self.env_keys[0]]
            except KeyError:
                raise ValueError("A valid API key must be passed when a"
                                 " SpeechRecognitionAPIConverter is initialized.")
        self.recognizer = sr.Recognizer()
        self.api_key = api_key

    def _convert(self, audio):
        import speech_recognition as sr

        if audio.filename is None:
            file = tempfile.mktemp() + '.wav'
            audio.clip.write_audiofile(file)
        else:
            file = audio.filename

        with sr.AudioFile(file) as source:
            clip = self.recognizer.record(source)
        text = getattr(self.recognizer, self.recognize_method)(
            clip, self.api_key)

        if audio.filename is None:
            os.remove(file)

        return ComplexTextStim(text=text)


class WitTranscriptionConverter(SpeechRecognitionAPIConverter):

    ''' Speech-to-text transcription via the Wit.ai API. '''

    _env_keys = 'WIT_AI_API_KEY'
    recognize_method = 'recognize_wit'


class GoogleSpeechAPIConverter(SpeechRecognitionAPIConverter):

    ''' Speech-to-text transcription via the Google Cloud Speech API. '''

    _env_keys = 'GOOGLE_APPLICATION_CREDENTIALS'
    recognize_method = 'recognize_google_cloud'

    def _convert(self, audio):
        with open(self.api_key) as json_data:
            self.api_key = json_data.read()
        super(GoogleSpeechAPIConverter, self)._convert(audio)


class IBMSpeechAPIConverter(AudioToTextConverter, EnvironmentKeyMixin):

    ''' Uses the IBM Watson Text to Speech API to run speech-to-text
    transcription on an audio file.

    Args:
        username (str): API credential username. Must be passed explicitly
            or stored in the environment variable specified in the _env_keys
            field.
        password (str): API credential password. Must be passed explicitly
            or stored in the environment variable specified in the _env_keys
            field.
        resolution (str): what resolution the resultant ComplexTextStim should
            be separated by (i.e. the unit each TextStim in the ComplexTextStim
            elements should be). Currently, only 'words' or 'phrases' are
            supported.
    '''

    _env_keys = ('IBM_USERNAME', 'IBM_PASSWORD')
    _log_attributes = ('resolution',)

    def __init__(self, username=None, password=None, resolution='words'):
        super(IBMSpeechAPIConverter, self).__init__()
        import speech_recognition as sr
        if username is None or password is None:
            try:
                username = os.environ['IBM_USERNAME']
                password = os.environ['IBM_PASSWORD']
            except KeyError:
                raise ValueError("A valid API key must be passed when a "
                                 "SpeechRecognitionConverter is initialized.")
        self.recognizer = sr.Recognizer()
        self.username = username
        self.password = password
        self.resolution = resolution

    def _convert(self, audio):
        import speech_recognition as sr

        if audio.filename is None:
            file = tempfile.mktemp() + '.wav'
            audio.clip.write_audiofile(file)
        else:
            file = audio.filename

        with sr.AudioFile(file) as source:
            clip = self.recognizer.record(source)

        _json = self._query_api(clip)
        if 'results' in _json:
            results = _json['results']
        else:
            raise Exception(
                'received invalid results from API: {0}'.format(str(_json)))
        elements = []
        for result in results:
            if result['final'] is True:
                timestamps = result['alternatives'][0]['timestamps']
                if self.resolution is 'words':
                    for entry in timestamps:
                        text = entry[0]
                        start = entry[1]
                        end = entry[2]
                        elements.append(TextStim(text=text, onset=start,
                                                 duration=end-start))
                elif self.resolution is 'phrases':
                    text = result['alternatives'][0]['transcript']
                    start = timestamps[0][1]
                    end = timestamps[-1][2]
                    elements.append(TextStim(text=text, onset=start,
                                             duration=end-start))
        return ComplexTextStim(elements=elements)

    def _query_api(self, clip):
        # Adapted from SpeechRecognition source code, modified to get text
        # onsets
        flac_data = clip.get_flac_data(
            convert_rate=None if clip.sample_rate >= 16000 else 16000,
            convert_width=None if clip.sample_width >= 2 else 2
        )
        model = "{0}_BroadbandModel".format("en-US")
        url = "https://stream.watsonplatform.net/speech-to-text/api/v1/recognize?{0}".format(urlencode({
            "profanity_filter": "false",
            "continuous": "true",
            "model": model,
            "timestamps": "true",
            "inactivity_timeout": -1,
        }))
        request = Request(url, data=flac_data, headers={
            "Content-Type": "audio/x-flac",
            "X-Watson-Learning-Opt-Out": "true",
        })

        if hasattr("", "encode"):  # Python 2.6 compatibility
            authorization_value = base64.standard_b64encode(
                "{0}:{1}".format(self.username, self.password).encode("utf-8")).decode("utf-8")
        else:
            authorization_value = base64.standard_b64encode(
                "{0}:{1}".format(self.username, self.password))
        request.add_header(
            "Authorization", "Basic {0}".format(authorization_value))

        try:
            response = urlopen(request, timeout=None)
        except HTTPError as e:
            raise Exception("recognition request failed: {0}".format(
                getattr(e, "reason", "status {0}".format(e.code))))
        except URLError as e:
            raise Exception(
                "recognition connection failed: {0}".format(e.reason))

        response_text = response.read().decode("utf-8")
        result = json.loads(response_text)
        return result
