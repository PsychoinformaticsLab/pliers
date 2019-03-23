''' IBM API-based Converter classes '''

import os
import base64
import json
import logging
from pliers.stimuli.text import TextStim, ComplexTextStim
from pliers.utils import attempt_to_import, verify_dependencies
from pliers.converters.audio import AudioToTextConverter
from pliers.transformers.api import APITransformer
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

sr = attempt_to_import('speech_recognition', 'sr')


class IBMSpeechAPIConverter(APITransformer, AudioToTextConverter):

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
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
        model (str): The model to use for speech recognition (e.g., 'en-US',
            'zh-CN', etc.). Don't include the "_BroadbandModel" suffix. 
    '''

    _env_keys = ('IBM_USERNAME', 'IBM_PASSWORD')
    _log_attributes = ('username', 'password', 'resolution', 'model')
    VERSION = '1.0'

    def __init__(self, username=None, password=None, resolution='words',
                 rate_limit=None, model='en-US'):
        verify_dependencies(['sr'])
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
        self.model = model
        super(IBMSpeechAPIConverter, self).__init__(rate_limit=rate_limit)

    @property
    def api_keys(self):
        return [self.username, self.password]

    def check_valid_keys(self):
        url = "https://stream.watsonplatform.net/speech-to-text/api/v1/recognize"
        request = Request(url)
        try:
            self._send_request(request)
            return True
        except Exception as e:
            if 'Not Authorized' in str(e):
                logging.warn(str(e))
                return False
            else:
                raise e

    def _convert(self, audio):
        verify_dependencies(['sr'])
        offset = 0.0 if audio.onset is None else audio.onset

        with audio.get_filename() as filename:
            with sr.AudioFile(filename) as source:
                clip = self.recognizer.record(source)

        _json = self._query_api(clip)
        if 'results' in _json:
            results = _json['results']
        else:
            raise Exception(
                'received invalid results from API: {0}'.format(str(_json)))
        elements = []
        order = 0
        for result in results:
            if result['final'] is True:
                timestamps = result['alternatives'][0]['timestamps']
                if self.resolution is 'words':
                    for entry in timestamps:
                        text = entry[0]
                        start = entry[1]
                        end = entry[2]
                        elements.append(TextStim(text=text,
                                                 onset=offset+start,
                                                 duration=end-start,
                                                 order=order))
                        order += 1
                elif self.resolution is 'phrases':
                    text = result['alternatives'][0]['transcript']
                    start = timestamps[0][1]
                    end = timestamps[-1][2]
                    elements.append(TextStim(text=text,
                                             onset=offset+start,
                                             duration=end-start,
                                             order=order))
                    order += 1
        return ComplexTextStim(elements=elements, onset=audio.onset)

    def _query_api(self, clip):
        # Adapted from SpeechRecognition source code, modified to get text
        # onsets
        flac_data = clip.get_flac_data(
            convert_rate=None if clip.sample_rate >= 16000 else 16000,
            convert_width=None if clip.sample_width >= 2 else 2
        )
        model = "{0}_BroadbandModel".format(self.model)
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
        return self._send_request(request)

    def _send_request(self, request):
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
