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
                               GoogleLanguageAPIExtractor,
                               GoogleLanguageAPIEntityExtractor,
                               GoogleLanguageAPISentimentExtractor,
                               GoogleLanguageAPISyntaxExtractor,
                               GoogleLanguageAPITextCategoryExtractor,
                               GoogleLanguageAPIEntitySentimentExtractor,
                               ExtractorResult,
                               merge_results)
from pliers.extractors.api.google import GoogleVisionAPIExtractor
from pliers.stimuli import ImageStim, VideoStim, TextStim
from pliers.utils import attempt_to_import, verify_dependencies
import pytest
import json
from os.path import join
from ...utils import get_test_data_path
import numpy as np

googleapiclient = attempt_to_import('googleapiclient', fromlist=['discovery'])

IMAGE_DIR = join(get_test_data_path(), 'image')
VIDEO_DIR = join(get_test_data_path(), 'video')
TEXT_DIR = join(get_test_data_path(), 'text')


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
    assert 'Apple' in result.columns
    assert result['Apple'][0] > 0.75

    url = 'https://via.placeholder.com/350x150'
    stim = ImageStim(url=url)
    result = ext.transform(stim).to_df()
    assert result['Text'][0] > 0.9

    ext = GoogleVisionAPILabelExtractor(discovery_file='nogood')
    assert not ext.validate_keys()


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_properties_extractor():
    ext = GoogleVisionAPIPropertyExtractor(num_retries=5)
    filename = join(get_test_data_path(), 'image', 'apple.jpg')
    stim = ImageStim(filename)
    result = ext.transform(stim).to_df()
    assert '158, 13, 29' in result.columns
    assert np.isfinite(result['158, 13, 29'][0])


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
    assert 'GoogleVisionAPILabelExtractor#Apple' in results.columns
    assert results.shape == (2, 32)

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
        assert result['shot_id'][0] == 0


@pytest.mark.long_test
@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_video_api_extractor2(caplog):
    segments = [{'startTimeOffset': '0.1s', 'endTimeOffset': '0.3s'},
                {'startTimeOffset': '0.3s', 'endTimeOffset': '0.45s'}]
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
        assert result['onset'][0] > 0.1 and result['onset'][0] < 0.3
        assert result['onset'][1] > 0.3 and result['onset'][1] < 0.45
        assert 'UNLIKELY' in result['pornographyLikelihood'][0]


@pytest.mark.long_test
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
        assert result.shape[1] > 20
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
        assert result.shape[1] > 10
        assert result['onset'][1] == 0.0
        assert np.isclose(result['onset'][2], 3.2, 0.1)
        assert np.isnan(result['cat'][1])
        assert result['cat'][2] > 0.5
        assert np.isnan(result['clock'][2])
        assert result['clock'][1] > 0.5 or result['clock'][0] > 0.5


@pytest.mark.long_test
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
        assert 'shot_id' in result.columns
        assert result['shot_id'][0] == 0

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
        assert 'shot_id' in result.columns
        assert result['shot_id'][1] == 1


@pytest.mark.long_test
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


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_language_api_extractor():
    verify_dependencies(['googleapiclient'])
    ext = GoogleLanguageAPIExtractor(features=['classifyText',
                                               'extractEntities'])
    stim = TextStim(text='hello world')

    with pytest.raises(googleapiclient.errors.HttpError):
        # Should fail because too few tokens
        ext.transform(stim)

    stim = TextStim(join(TEXT_DIR, 'scandal.txt'))
    result = ext.transform(stim).to_df(timing=False, object_id='auto')
    assert result.shape == (43, 10)
    assert 'category_/Books & Literature' in result.columns
    assert result['category_/Books & Literature'][0] > 0.5
    irene = result[result['text'] == 'Irene Adler']
    assert (irene['type'] == 'PERSON').all()
    assert not irene['metadata_wikipedia_url'].isna().any()
    # Document row shouldn't have entity features, and vice versa
    assert np.isnan(result.iloc[0]['text'])
    assert np.isnan(result.iloc[1]['category_/Books & Literature']).all()


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_language_api_entity_extractor():
    verify_dependencies(['googleapiclient'])
    ext = GoogleLanguageAPIEntityExtractor()
    stim = TextStim(join(TEXT_DIR, 'sample_text_with_entities.txt'))
    result = ext.transform(stim).to_df(timing=False, object_id='auto')
    assert result.shape == (10, 9)
    assert result['text'][0] == 'Google'
    assert result['type'][0] == 'ORGANIZATION'
    assert result['salience'][0] > 0.0 and result['salience'][0] < 0.5
    assert result['begin_char_index'][4] == 165.0
    assert result['end_char_index'][4] == 172.0
    assert result['text'][4] == 'Android'
    assert result['type'][4] == 'CONSUMER_GOOD'


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_language_api_sentiment_extractor():
    verify_dependencies(['googleapiclient'])
    ext = GoogleLanguageAPISentimentExtractor()
    stim = TextStim(join(TEXT_DIR, 'scandal.txt'))
    result = ext.transform(stim).to_df(timing=False, object_id='auto')
    assert result.shape == (12, 7)
    assert 'sentiment_magnitude' in result.columns
    assert 'text' in result.columns
    doc_sentiment = result['sentiment_score'][11]
    assert doc_sentiment < 0.3 and doc_sentiment > -0.3
    assert result['begin_char_index'][7] == 565.0
    assert result['end_char_index'][7] == 672.0
    assert result['sentiment_magnitude'][7] > 0.6
    assert result['sentiment_score'][7] > 0.6


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_language_api_syntax_extractor():
    verify_dependencies(['googleapiclient'])
    ext = GoogleLanguageAPISyntaxExtractor()
    stim = TextStim(join(TEXT_DIR, 'sample_text_with_entities.txt'))
    result = ext.transform(stim).to_df(timing=False, object_id='auto')
    assert result.shape == (32, 20)
    his = result[result['text'] == 'his']
    assert (his['person'] == 'THIRD').all()
    assert (his['gender'] == 'MASCULINE').all()
    assert (his['case'] == 'GENITIVE').all()
    their = result[result['text'] == 'their']
    assert (their['person'] == 'THIRD').all()
    assert (their['number'] == 'PLURAL').all()
    love = result[result['text'] == 'love']
    assert (love['tag'] == 'VERB').all()
    assert (love['mood'] == 'INDICATIVE').all()
    headquartered = result[result['text'] == 'headquartered']
    assert (headquartered['tense'] == 'PAST').all()
    assert (headquartered['lemma'] == 'headquarter').all()
    google = result[result['text'] == 'Google']
    assert (google['proper'] == 'PROPER').all()
    assert (google['tag'] == 'NOUN').all()
    assert (google['dependency_label'] == 'NSUBJ').all()
    assert (google['dependency_headTokenIndex'] == 7).all()


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_language_api_category_extractor():
    verify_dependencies(['googleapiclient'])
    ext = GoogleLanguageAPITextCategoryExtractor()
    stim = TextStim(join(TEXT_DIR, 'sample_text_with_entities.txt'))
    result = ext.transform(stim).to_df(timing=False, object_id='auto')
    assert result.shape == (1, 4)
    assert 'category_/Computers & Electronics' in result.columns
    assert result['category_/Computers & Electronics'][0] > 0.3
    assert 'category_/News' in result.columns
    assert result['category_/News'][0] > 0.3
    assert result['language'][0] == 'en'


@pytest.mark.requires_payment
@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_language_api_entity_sentiment_extractor():
    verify_dependencies(['googleapiclient'])
    ext = GoogleLanguageAPIEntitySentimentExtractor()
    stim = TextStim(join(TEXT_DIR, 'sample_text_with_entities.txt'))
    result = ext.transform(stim).to_df(timing=False, object_id='auto')
    # Produces same result as entity extractor with sentiment columns
    assert result.shape == (10, 11)
    assert result['text'][8] == 'phones'
    assert result['type'][8] == 'CONSUMER_GOOD'
    assert 'sentiment_score' in result.columns
    assert result['sentiment_score'][8] > 0.6 # 'love their ... phones'
