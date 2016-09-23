from featurex.extractors.google import (GoogleVisionAPIExtractor,
                                        GoogleVisionAPIFaceExtractor)
from featurex.stimuli.image import ImageStim
from featurex import Value
import pytest
import os
import json
from os.path import join
from .utils import get_test_data_path
import numpy as np


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_extractor_inits():
    ext = GoogleVisionAPIExtractor(num_retries=5)
    assert ext.num_retries == 5
    assert ext.max_results == 100
    assert ext.service is not None

@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api__face_extractor_inits():
    ext = GoogleVisionAPIFaceExtractor(num_retries=5)
    assert ext.num_retries == 5
    assert ext.max_results == 100
    assert ext.service is not None

    # Load dummy stim
    filename = join(get_test_data_path(), 'image', 'obama.jpg')
    stim = ImageStim(filename)

    # Test parsing of individual response
    filename = join(get_test_data_path(), 'payloads', 'google_vision_api_face_payload.json')
    response = json.load(open(filename, 'r'))
    result = ext._parse_response(stim, response['faceAnnotations'][0])
    assert isinstance(result, Value)
    assert result.stim == stim
    assert result.data['angerLikelihood'] == 'VERY_UNLIKELY'
    assert result.data['landmark_LEFT_EYE_BOTTOM_BOUNDARY_y'] == 257.023
    assert np.isnan(result.data['boundingPoly_vertex2_y'])
    values = stim.extract([ext]).data['GoogleVisionAPIFaceExtractor']
    assert isinstance(values, Value)
    assert values.data['rollAngle'] == -1.6712251
    assert values.data['angerLikelihood'] == "VERY_UNLIKELY"
