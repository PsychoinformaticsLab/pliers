from os.path import join
import os
from ..utils import get_test_data_path
from pliers.extractors import (IndicoAPITextExtractor,
                               IndicoAPIImageExtractor,
                               ClarifaiAPIExtractor)
from pliers.stimuli import (TextStim, ComplexTextStim, ImageStim)
from pliers.extractors.base import merge_results
import numpy as np
import pytest


@pytest.mark.skipif("'INDICO_APP_KEY' not in os.environ")
def test_indico_api_text_extractor():

    ext = IndicoAPITextExtractor(api_key=os.environ['INDICO_APP_KEY'],
                                 models=['emotion', 'personality'])

    # With ComplexTextStim input
    srtfile = join(get_test_data_path(), 'text', 'wonderful.srt')
    srt_stim = ComplexTextStim(srtfile, onset=4.2)
    result = merge_results(ext.transform(srt_stim), extractor_names=False)
    outdfKeysCheck = {
        'onset',
        'duration',
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


@pytest.mark.skipif("'INDICO_APP_KEY' not in os.environ")
def test_indico_api_image_extractor():

    ext = IndicoAPIImageExtractor(api_key=os.environ['INDICO_APP_KEY'],
                                  models=['fer', 'content_filtering'])

    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    result1 = merge_results(ext.transform([stim1, stim1]), extractor_names=False)
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
                    'duration'}

    assert set(result1.columns) - set(['stim_name']) == outdfKeysCheck | meta_columns
    assert result1['content_filtering'][0] < 0.2

    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    result2 = ext.transform(stim2).to_df(timing=False, object_id=True)
    assert set(result2.columns) == outdfKeysCheck
    assert result2['fer_Happy'][0] > 0.7


@pytest.mark.skipif("'CLARIFAI_API_KEY' not in os.environ")
def test_clarifai_api_extractor():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    result = ClarifaiAPIExtractor().transform(stim).to_df()
    assert result['apple'][0] > 0.5
    assert result.ix[:, 5][0] > 0.0

    result = ClarifaiAPIExtractor(max_concepts=5).transform(stim).to_df()
    assert result.shape == (1, 8)

    result = ClarifaiAPIExtractor(min_value=0.9).transform(stim).to_df()
    assert all(np.isnan(d) or d > 0.9 for d in result.values[0, 3:])

    concepts = ['cat', 'dog']
    result = ClarifaiAPIExtractor(select_concepts=concepts).transform(stim)
    result = result.to_df()
    assert result.shape == (1, 5)
    assert 'cat' in result.columns and 'dog' in result.columns


@pytest.mark.skipif("'CLARIFAI_API_KEY' not in os.environ")
def test_clarifai_api_extractor_batch():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    ext = ClarifaiAPIExtractor()
    results = ext.transform([stim, stim2])
    results = merge_results(results)
    assert results['ClarifaiAPIExtractor#apple'][0] > 0.5 or \
        results['ClarifaiAPIExtractor#apple'][1] > 0.5

    # This takes too long to execute
    # video = VideoStim(join(get_test_data_path(), 'video', 'small.mp4'))
    # results = ExtractorResult.merge_stims(ext.transform(video))
    # assert 'Lego' in results.columns and 'robot' in results.columns
