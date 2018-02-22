from os.path import join
from ...utils import get_test_data_path
from pliers.extractors import ClarifaiAPIExtractor
from pliers.stimuli import ImageStim
from pliers.extractors.base import merge_results
import numpy as np
import pytest


@pytest.mark.skipif("'CLARIFAI_API_KEY' not in os.environ")
def test_clarifai_api_extractor():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    result = ClarifaiAPIExtractor().transform(stim).to_df()
    assert result['apple'][0] > 0.5
    assert result.ix[:, 5][0] > 0.0

    result = ClarifaiAPIExtractor(max_concepts=5).transform(stim).to_df()
    assert result.shape == (1, 9)

    result = ClarifaiAPIExtractor(
        min_value=0.9).transform(stim).to_df(object_id=False)
    assert all(np.isnan(d) or d > 0.9 for d in result.values[0, 3:])

    concepts = ['cat', 'dog']
    result = ClarifaiAPIExtractor(select_concepts=concepts).transform(stim)
    result = result.to_df()
    assert result.shape == (1, 6)
    assert 'cat' in result.columns and 'dog' in result.columns

    url = 'https://tuition.utexas.edu/sites/all/themes/tuition/logo.png'
    stim = ImageStim(url=url)
    result = ClarifaiAPIExtractor(max_concepts=5).transform(stim).to_df()
    assert result.shape == (1, 9)
    assert result['symbol'][0] > 0.8


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
