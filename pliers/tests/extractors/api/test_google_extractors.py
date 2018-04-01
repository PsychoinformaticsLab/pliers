from pliers import config
from pliers.filters import FrameSamplingFilter
from pliers.extractors import (GoogleVisionAPIFaceExtractor,
                               GoogleVisionAPILabelExtractor,
                               GoogleVisionAPIPropertyExtractor,
                               GoogleVisionAPISafeSearchExtractor,
                               GoogleVisionAPIWebEntitiesExtractor,
                               ExtractorResult,
                               merge_results)
from pliers.extractors.api.google import GoogleVisionAPIExtractor
from pliers.stimuli import ImageStim, VideoStim
import pytest
import json
from os.path import join
from ...utils import get_test_data_path
import numpy as np


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_extractor_inits():
    ext = GoogleVisionAPIExtractor(num_retries=5)
    assert ext.num_retries == 5
    assert ext.max_results == 100
    assert ext.service is not None


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_face_extractor_inits():
    ext = GoogleVisionAPIFaceExtractor(num_retries=5)
    assert ext.num_retries == 5
    assert ext.max_results == 100
    assert ext.service is not None

    # Test parsing of individual response
    filename = join(
        get_test_data_path(), 'payloads', 'google_vision_api_face_payload.json')
    response = json.load(open(filename, 'r'))
    stim = ImageStim(join(get_test_data_path(), 'image', 'obama.jpg'))
    res = ExtractorResult(response['faceAnnotations'], stim, ext)
    df = res.to_df()
    assert df['angerLikelihood'][0] == 'VERY_UNLIKELY'
    assert df['landmark_LEFT_EYE_BOTTOM_BOUNDARY_y'][0] == 257.023
    assert np.isnan(df['boundingPoly_vertex2_y'][0])


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_face_extractor():
    ext = GoogleVisionAPIFaceExtractor(num_retries=5)
    assert ext.validate_keys()
    filename = join(get_test_data_path(), 'image', 'obama.jpg')
    stim = ImageStim(filename)
    result = ext.transform(stim).to_df()
    assert 'joyLikelihood' in result.columns
    assert result['joyLikelihood'][0] == 'VERY_LIKELY'
    assert float(result['face_detectionConfidence'][0]) > 0.7

    ext = GoogleVisionAPIFaceExtractor(discovery_file='nogood')
    assert not ext.validate_keys()


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_multiple_face_extraction():
    filename = join(get_test_data_path(), 'image', 'thai_people.jpg')
    stim = ImageStim(filename)
    # Only first record
    ext = GoogleVisionAPIFaceExtractor()
    result1 = ext.transform(stim).to_df(handle_annotations='first')
    assert 'joyLikelihood' in result1.columns
    # All records
    ext = GoogleVisionAPIFaceExtractor()
    result2 = ext.transform(stim).to_df()
    assert 'joyLikelihood' in result2.columns
    assert result2.shape[0] > result1.shape[0]


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_face_batch():
    stims = ['apple', 'obama', 'thai_people']
    stim_files = [join(get_test_data_path(), 'image', '%s.jpg' % s)
                  for s in stims]
    stims = [ImageStim(s) for s in stim_files]
    ext = GoogleVisionAPIFaceExtractor()
    result = ext.transform(stims)
    result = merge_results(result, format='wide', extractor_names=False,
                           handle_annotations='first')
    assert result.shape == (2, 139)
    assert 'joyLikelihood' in result.columns
    assert result['joyLikelihood'][0] == 'VERY_LIKELY'
    assert result['joyLikelihood'][1] == 'VERY_LIKELY'

    video = VideoStim(join(get_test_data_path(), 'video', 'obama_speech.mp4'))
    conv = FrameSamplingFilter(every=10)
    video = conv.transform(video)
    result = ext.transform(video)
    result = merge_results(result, format='wide', extractor_names=False)
    assert 'joyLikelihood' in result.columns
    assert result.shape == (22, 139)

    video = VideoStim(join(get_test_data_path(), 'video', 'small.mp4'))
    video = conv.transform(video)
    result = ext.transform(video)
    result = merge_results(result, format='wide', extractor_names=False)
    assert 'joyLikelihood' not in result.columns
    assert len(result) == 0


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_label_extractor():
    ext = GoogleVisionAPILabelExtractor(num_retries=5)
    assert ext.validate_keys()
    filename = join(get_test_data_path(), 'image', 'apple.jpg')
    stim = ImageStim(filename)
    result = ext.transform(stim).to_df()
    assert 'apple' in result.columns
    assert result['apple'][0] > 0.75

    url = 'https://tuition.utexas.edu/sites/all/themes/tuition/logo.png'
    stim = ImageStim(url=url)
    result = ext.transform(stim).to_df()
    assert result['orange'][0] > 0.7

    ext = GoogleVisionAPILabelExtractor(discovery_file='nogood')
    assert not ext.validate_keys()


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_properties_extractor():
    ext = GoogleVisionAPIPropertyExtractor(num_retries=5)
    filename = join(get_test_data_path(), 'image', 'apple.jpg')
    stim = ImageStim(filename)
    result = ext.transform(stim).to_df()
    assert (158, 13, 29) in result.columns
    assert np.isfinite(result[(158, 13, 29)][0])


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_safe_search():
    ext = GoogleVisionAPISafeSearchExtractor(num_retries=5)
    filename = join(get_test_data_path(), 'image', 'obama.jpg')
    stim = ImageStim(filename)
    result = ext.transform(stim).to_df()
    assert 'adult' in result.columns
    assert result['violence'][0] == 'VERY_UNLIKELY'


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_web_entities():
    ext = GoogleVisionAPIWebEntitiesExtractor(num_retries=5)
    filename = join(get_test_data_path(), 'image', 'obama.jpg')
    stim = ImageStim(filename)
    result = ext.transform(stim).to_df()
    assert 'Barack Obama' in result.columns


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_extractor_large():
    default = config.get_option('allow_large_jobs')
    default_large = config.get_option('large_job')
    config.set_option('allow_large_jobs', False)
    config.set_option('large_job', 1)

    ext = GoogleVisionAPILabelExtractor()
    images = [ImageStim(join(get_test_data_path(), 'image', 'apple.jpg'))] * 2
    with pytest.raises(ValueError):
        merge_results(ext.transform(images))

    config.set_option('allow_large_jobs', True)
    results = merge_results(ext.transform(images))
    assert 'GoogleVisionAPILabelExtractor#apple' in results.columns
    assert results.shape == (1, 16)  # not 2 cause all the same instance

    config.set_option('allow_large_jobs', default)
    config.set_option('large_job', default_large)
