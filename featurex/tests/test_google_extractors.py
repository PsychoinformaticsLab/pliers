from featurex.extractors.google import (GoogleVisionAPIExtractor,
                                        GoogleVisionAPIFaceExtractor,
                                        GoogleVisionAPITextExtractor,
                                        GoogleVisionAPILabelExtractor,
                                        GoogleVisionAPIPropertyExtractor)
from featurex.stimuli.image import ImageStim
from featurex import Value
from featurex import Event
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
def test_google_vision_api_face_extractor_inits():
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
    results = ext._parse_annotations(stim, response['faceAnnotations'])
    result = results[0]
    assert isinstance(result, Value)
    assert result.stim == stim
    assert result.data['angerLikelihood'] == 'VERY_UNLIKELY'
    assert result.data['landmark_LEFT_EYE_BOTTOM_BOUNDARY_y'] == 257.023
    assert np.isnan(result.data['boundingPoly_vertex2_y'])
    values = stim.extract([ext]).data['GoogleVisionAPIFaceExtractor'][0]
    assert isinstance(values, Value)
    assert values.data['rollAngle'] == -1.6712251
    assert values.data['angerLikelihood'] == "VERY_UNLIKELY"

@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_text_extractor():
    ext = GoogleVisionAPITextExtractor(num_retries=5)
    filename = join(_get_test_data_path(), 'image', 'button.jpg')
    stim = ImageStim(filename)
    values = ext.apply(stim)[0]
    assert values.data['locale'] == 'en'
    assert 'Exit' in values.data['description']
    assert np.isfinite(values.data['boundingPoly_vertex2_y'])

@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_label_extractor():
    ext = GoogleVisionAPILabelExtractor(num_retries=5)
    filename = join(_get_test_data_path(), 'image', 'apple.jpg')
    stim = ImageStim(filename)
    values = ext.apply(stim)[0]
    assert values.data['description'] == 'apple'
    assert values.data['score'] > 0.75

@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_properties_extractor():
    ext = GoogleVisionAPIPropertyExtractor(num_retries=5)
    filename = join(_get_test_data_path(), 'image', 'apple.jpg')
    stim = ImageStim(filename)
    values = ext.apply(stim)[0]
    assert 'dominantColors' in values.data
    assert np.isfinite(values.data['dominantColors']['colors'][0]['score'])
    assert np.isfinite(values.data['dominantColors']['colors'][0]['pixelFraction'])

@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_extractor_multi_stim():
    ext = GoogleVisionAPITextExtractor(num_retries=5)
    filename = join(_get_test_data_path(), 'image', 'button.jpg')
    stim = ImageStim(filename)
    filename2 = join(_get_test_data_path(), 'image', 'apple.jpg')
    stim2 = ImageStim(filename2)
    events = ext.apply([stim, stim2])
    assert isinstance(events[0], Event)
    assert len(events) == 2
    assert not events[1].values

