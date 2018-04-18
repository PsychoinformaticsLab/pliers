from pliers import config
from pliers.filters import FrameSamplingFilter
from pliers.extractors import (GoogleVisionAPIFaceExtractor,
                               GoogleVisionAPILabelExtractor,
                               GoogleVisionAPIPropertyExtractor,
                               GoogleVisionAPISafeSearchExtractor,
                               GoogleVisionAPIWebEntitiesExtractor,
                               GoogleVideoIntelligenceAPIExtractor,
                               GoogleVideoAPILabelDetectionExtractor,
                               GoogleVideoAPIShotDetectionExtractor,
                               GoogleVideoAPIExplicitDetectionExtractor,
                               ExtractorResult,
                               merge_results)
from pliers.extractors.api.google import GoogleVisionAPIExtractor
from pliers.stimuli import ImageStim, VideoStim
import pytest
import json
from os.path import join
from ...utils import get_test_data_path
import numpy as np

IMAGE_DIR = join(get_test_data_path(), 'image')
VIDEO_DIR = join(get_test_data_path(), 'video')


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
    ext = GoogleVisionAPIFaceExtractor(batch_size=5)
    result = ext.transform(stims)
    result = merge_results(result, format='wide', extractor_names=False,
                           handle_annotations='first')
    assert result.shape == (2, 139)
    assert 'joyLikelihood' in result.columns
    assert result['joyLikelihood'][0] == 'VERY_LIKELY'
    assert result['joyLikelihood'][1] == 'VERY_LIKELY'

    video = VideoStim(join(VIDEO_DIR, 'obama_speech.mp4'))
    conv = FrameSamplingFilter(every=10)
    video = conv.transform(video)
    result = ext.transform(video)
    result = merge_results(result, format='wide', extractor_names=False)
    assert 'joyLikelihood' in result.columns
    assert result.shape == (22, 139)

    video = VideoStim(join(VIDEO_DIR, 'small.mp4'))
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
    default_cache = config.get_option('cache_transformers')
    config.set_option('allow_large_jobs', False)
    config.set_option('large_job', 1)
    config.set_option('cache_transformers', False)

    ext = GoogleVisionAPILabelExtractor()
    images = [ImageStim(join(IMAGE_DIR, 'apple.jpg')),
              ImageStim(join(IMAGE_DIR, 'obama.jpg'))]
    with pytest.raises(ValueError):
        merge_results(ext.transform(images))

    config.set_option('allow_large_jobs', True)
    results = merge_results(ext.transform(images))
    assert 'GoogleVisionAPILabelExtractor#apple' in results.columns
    assert results.shape == (2, 36)

    config.set_option('allow_large_jobs', default)
    config.set_option('large_job', default_large)
    config.set_option('cache_transformers', default_cache)


@pytest.mark.long_test
@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_video_api_extractor(caplog):
    ext = GoogleVideoIntelligenceAPIExtractor(timeout=1)
    stim = VideoStim(join(VIDEO_DIR, 'park.mp4'))
    result = ext.transform(stim)

    log_message = caplog.records[-1].message
    assert log_message == ("The extraction reached the timeout limit of %fs, "
                  "which means the API may not have finished analyzing the "
                  "video and the results may be empty or incomplete." % 1.0)

    ext = GoogleVideoIntelligenceAPIExtractor(timeout=500,
                                              features=['LABEL_DETECTION',
                                                    'SHOT_CHANGE_DETECTION'])
    result = ext.transform(stim).to_df()
    log_message = caplog.records[-1].message
    incomplete = (log_message == ("The extraction reached the timeout limit of"
                " %fs, which means the API may not have finished analyzing the"
                " video and the results may be empty or incomplete." % 500))
    if not incomplete:
        assert result.shape == (1, 31)
        assert result['onset'][0] == 0.0
        assert result['duration'][0] > 0.5 and result['duration'][0] < 0.6
        assert result['category_plant'][0] > 0.5
        assert result['park'][0] > 0.5
        assert result['shot'][0] == 1.0


@pytest.mark.long_test
@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_video_api_extractor2(caplog):
    segments = [{'startTimeOffset': '0.1s', 'endTimeOffset': '0.2s'},
                {'startTimeOffset': '0.3s', 'endTimeOffset': '0.4s'}]
    ext = GoogleVideoIntelligenceAPIExtractor(timeout=500, segments=segments,
                                    features=['EXPLICIT_CONTENT_DETECTION'])
    stim = VideoStim(join(VIDEO_DIR, 'park.mp4'))
    result = ext.transform(stim).to_df()
    log_message = caplog.records[-1].message
    incomplete = (log_message == ("The extraction reached the timeout limit of"
                " %fs, which means the API may not have finished analyzing the"
                " video and the results may be empty or incomplete." % 500))
    if not incomplete:
        assert result.shape == (2, 5)
        assert result['onset'][0] == 0.1
        assert result['onset'][1] == 0.3
        assert 'UNLIKELY' in result['pornographyLikelihood'][0]


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_video_api_label_extractor(caplog):
    ext = GoogleVideoAPILabelDetectionExtractor(mode='FRAME_MODE',
                                                stationary_camera=True)
    stim = VideoStim(join(VIDEO_DIR, 'small.mp4'))
    ex_result = ext.transform(stim)
    log_message = caplog.records[-1].message
    incomplete = (log_message == ("The extraction reached the timeout limit of"
            " %fs, which means the API may not have finished analyzing the"
            " video and the results may be empty or incomplete." % 90))
    if not incomplete:
        result = ex_result.to_df()
        assert result.shape == (7, 25)
        assert 'category_toy' in result.columns
        assert result['toy'][0] > 0.5
        assert np.isclose(result['duration'][0], stim.duration, 0.1)
        result = ex_result.to_df(format='long')
        assert 'pornographyLikelihood' not in result['feature']
        assert np.nan not in result['value']

    ext = GoogleVideoAPILabelDetectionExtractor(mode='SHOT_MODE')
    stim = VideoStim(join(VIDEO_DIR, 'shot_change.mp4'))
    ex_result = ext.transform(stim)
    log_message = caplog.records[-1].message
    incomplete = (log_message == ("The extraction reached the timeout limit of"
            " %fs, which means the API may not have finished analyzing the"
            " video and the results may be empty or incomplete." % 90))
    if not incomplete:
        raw = ex_result.raw['response']['annotationResults'][0]
        assert 'shotLabelAnnotations' in raw
        result = ex_result.to_df()
        assert result.shape == (3, 17)
        assert result['onset'][1] == 0.0
        assert np.isclose(result['onset'][2], 3.2, 0.1)
        assert np.isnan(result['cat'][1])
        assert result['cat'][2] > 0.5
        assert np.isnan(result['clock'][2])
        assert result['clock'][1] > 0.5


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_video_api_shot_extractor(caplog):
    ext = GoogleVideoAPIShotDetectionExtractor(request_rate=3)
    stim = VideoStim(join(VIDEO_DIR, 'small.mp4'))
    result = ext.transform(stim).to_df()
    log_message = caplog.records[-1].message
    incomplete = (log_message == ("The extraction reached the timeout limit of"
                " %fs, which means the API may not have finished analyzing the"
                " video and the results may be empty or incomplete." % 90))
    if not incomplete:
        assert result.shape == (1, 5)
        assert result['onset'][0] == 0.0
        assert np.isclose(result['duration'][0], stim.duration, 0.1)
        assert 'shot' in result.columns
        assert result['shot'][0] == 1.0

    ext = GoogleVideoAPIShotDetectionExtractor()
    stim = VideoStim(join(VIDEO_DIR, 'shot_change.mp4'))
    result = ext.transform(stim).to_df()
    log_message = caplog.records[-1].message
    incomplete = (log_message == ("The extraction reached the timeout limit of"
                " %fs, which means the API may not have finished analyzing the"
                " video and the results may be empty or incomplete." % 90))
    if not incomplete:
        assert result.shape == (2, 5)
        assert np.isclose(result['onset'][1], 3.2, 0.1)
        assert 'shot' in result.columns
        assert result['shot'][1] == 1.0


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_video_api_explicit_extractor(caplog):
    ext = GoogleVideoAPIExplicitDetectionExtractor(request_rate=3)
    stim = VideoStim(join(VIDEO_DIR, 'small.mp4'), onset=4.2)
    result = ext.transform(stim).to_df()
    log_message = caplog.records[-1].message
    incomplete = (log_message == ("The extraction reached the timeout limit of"
                " %fs, which means the API may not have finished analyzing the"
                " video and the results may be empty or incomplete." % 90))
    if not incomplete:
        assert result.shape[1] == 5
        assert result['onset'][0] >= 4.2
        assert 'pornographyLikelihood' in result.columns
        assert 'UNLIKELY' in result['pornographyLikelihood'][0]
