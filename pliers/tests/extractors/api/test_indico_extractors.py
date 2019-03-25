from os.path import join
import os
from ...utils import get_test_data_path
from pliers import config
from pliers.extractors import (IndicoAPITextExtractor,
                               IndicoAPIImageExtractor)
from pliers.stimuli import (TextStim, ComplexTextStim, ImageStim)
from pliers.extractors.base import merge_results
import pytest
import time

IMAGE_DIR = join(get_test_data_path(), 'image')


@pytest.mark.requires_payment
@pytest.mark.skipif("'INDICO_APP_KEY' not in os.environ")
def test_indico_api_text_extractor():

    ext = IndicoAPITextExtractor(api_key=os.environ['INDICO_APP_KEY'],
                                 models=['emotion', 'personality'])
    assert ext.validate_keys()

    # With ComplexTextStim input
    srtfile = join(get_test_data_path(), 'text', 'wonderful.srt')
    srt_stim = ComplexTextStim(srtfile, onset=4.2)
    result = merge_results(ext.transform(srt_stim), extractor_names=False)
    outdfKeysCheck = {
        'onset',
        'duration',
        'order',
        'object_id',
        'emotion_anger',
        'emotion_fear',
        'emotion_joy',
        'emotion_sadness',
        'emotion_surprise',
        'personality_openness',
        'personality_extraversion',
        'personality_agreeableness',
        'personality_conscientiousness'}
    meta_columns = {'source_file',
                    'history',
                    'class',
                    'filename'}
    assert set(result.columns) - set(['stim_name']) == outdfKeysCheck | meta_columns
    assert result['onset'][1] == 92.622

    # With TextStim input
    ts = TextStim(text="It's a wonderful life.")
    result = ext.transform(ts).to_df(object_id=True)
    assert set(result.columns) == outdfKeysCheck
    assert len(result) == 1

    ext = IndicoAPITextExtractor(api_key='nogood', models=['language'])
    assert not ext.validate_keys()


@pytest.mark.requires_payment
@pytest.mark.skipif("'INDICO_APP_KEY' not in os.environ")
def test_indico_api_image_extractor():
    ext = IndicoAPIImageExtractor(api_key=os.environ['INDICO_APP_KEY'],
                                  models=['fer', 'content_filtering'])
    stim1 = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    result1 = merge_results(ext.transform([stim1, stim1]),
                            extractor_names=False)
    outdfKeysCheck = {
        'object_id',
        'fer_Surprise',
        'fer_Neutral',
        'fer_Sad',
        'fer_Happy',
        'fer_Angry',
        'fer_Fear',
        'content_filtering'}
    meta_columns = {'source_file',
                    'history',
                    'class',
                    'filename',
                    'onset',
                    'duration',
                    'order'}

    assert set(result1.columns) - set(['stim_name']) == outdfKeysCheck | meta_columns
    assert result1['content_filtering'][0] < 0.2

    stim2 = ImageStim(join(IMAGE_DIR, 'obama.jpg'))
    result2 = ext.transform(stim2).to_df(timing=False, object_id=True)
    assert set(result2.columns) == outdfKeysCheck
    assert result2['fer_Happy'][0] > 0.7

    url = 'https://via.placeholder.com/350x150'
    stim = ImageStim(url=url)
    result = ext.transform(stim).to_df()
    assert result['fer_Neutral'][0] > 0.


@pytest.mark.requires_payment
@pytest.mark.skipif("'INDICO_APP_KEY' not in os.environ")
def test_indico_api_extractor_large():
    default = config.get_option('allow_large_jobs')
    default_large = config.get_option('large_job')
    config.set_option('allow_large_jobs', False)
    config.set_option('large_job', 1)

    ext = IndicoAPIImageExtractor(models=['fer'])

    images = [ImageStim(join(IMAGE_DIR, 'apple.jpg')),
              ImageStim(join(IMAGE_DIR, 'obama.jpg'))]
    with pytest.raises(ValueError):
        merge_results(ext.transform(images))

    config.set_option('allow_large_jobs', True)

    results = merge_results(ext.transform(images))
    assert 'IndicoAPIImageExtractor#fer_Neutral' in results.columns
    assert results.shape == (2, 15)

    config.set_option('allow_large_jobs', default)
    config.set_option('large_job', default_large)


@pytest.mark.requires_payment
@pytest.mark.skipif("'INDICO_APP_KEY' not in os.environ")
def test_indico_api_extractor_rate_limit():
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    stim2 = ImageStim(join(IMAGE_DIR, 'obama.jpg'))
    ext = IndicoAPIImageExtractor(models=['image_features'], rate_limit=5, batch_size=1)
    t1 = time.time()
    ext.transform([stim, stim2])
    t2 = time.time()
    assert t2 - t1 >= 5
