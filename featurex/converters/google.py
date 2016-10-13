import argparse
import base64
import os
import re
import sys
from featurex.converters.image import ImageToTextConverter
from featurex.converters import Converter
from featurex.stimuli.image import ImageStim
from featurex.stimuli.text import TextStim
from featurex.google import GoogleVisionAPITransformer
from featurex import Value, Event
import tempfile
from scipy.misc import imsave
import numpy as np

try:
    from googleapiclient import discovery, errors
    from oauth2client.client import GoogleCredentials
except ImportError:
    pass


class GoogleVisionAPITextConverter(GoogleVisionAPITransformer, ImageToTextConverter):

    request_type = 'TEXT_DETECTION'
    response_object = 'textAnnotations'

    def _convert(self, stim):
        request =  self._build_request([stim])
        responses = self._query_api(request)
        response = responses[0]

        if response:
            annotations = response[self.response_object]
            # Concatenate all the annotation
            text = ''
            for annotation in annotations:
                text += annotation['description']
            return TextStim(text=text)
        else:
            return TextStim(text='')
