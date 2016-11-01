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


    def __init__(self, handle_annotations='first', **kwargs):
        super(GoogleVisionAPITextConverter, self).__init__(**kwargs)
        self.handle_annotations = handle_annotations


    def _convert(self, stim):
        request =  self._build_request([stim])
        responses = self._query_api(request)
        response = responses[0]

        if response:
            annotations = response[self.response_object]
            # Combine the annotations
            if self.handle_annotations == 'first':
                text = annotations[0]['description']
                return TextStim(text=text)
            elif self.handle_annotations == 'concatenate':
                text = ''
                for annotation in annotations:
                    text += annotation['description']
                return TextStim(text=text)
            elif self.handle_annotations == 'list':
                texts = []
                for annotation in annotations:
                    texts.append(TextStim(text=annotation['description']))
                return texts
            
        else:
            return TextStim(text='')
