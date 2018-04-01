from os.path import join
from ...utils import get_test_data_path
from pliers import config
from pliers.extractors import ClarifaiAPIExtractor
from pliers.extractors.base import merge_results
from pliers.stimuli import ImageStim, VideoStim
import numpy as np
import pytest
import time

IMAGE_DIR = join(get_test_data_path(), 'image')


@pytest.mark.skipif("'CLARIFAI_API_KEY' not in os.environ")
def test_clarifai_api_extractor():
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    ext = ClarifaiAPIExtractor()
    assert ext.validate_keys()
    result = ext.transform(stim).to_df()
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

    ext = ClarifaiAPIExtractor(api_key='nogood')
    assert not ext.validate_keys()


@pytest.mark.skipif("'CLARIFAI_API_KEY' not in os.environ")
def test_clarifai_api_extractor_batch():
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    stim2 = ImageStim(join(IMAGE_DIR, 'obama.jpg'))
    ext = ClarifaiAPIExtractor()
    results = ext.transform([stim, stim2])
    results = merge_results(results)
    assert results['ClarifaiAPIExtractor#apple'][0] > 0.5 or \
        results['ClarifaiAPIExtractor#apple'][1] > 0.5


@pytest.mark.skipif("'CLARIFAI_API_KEY' not in os.environ")
def test_clarifai_api_extractor_large():
    default = config.get_option('allow_large_jobs')
    default_large = config.get_option('large_job')
    config.set_option('allow_large_jobs', False)
    config.set_option('large_job', 1)

    ext = ClarifaiAPIExtractor()
    images = [ImageStim(join(get_test_data_path(), 'image', 'apple.jpg'))] * 2
    with pytest.raises(ValueError):
        merge_results(ext.transform(images))

    config.set_option('allow_large_jobs', True)
    results = merge_results(ext.transform(images))
    assert 'ClarifaiAPIExtractor#apple' in results.columns
    assert results.shape == (1, 29)  # not 2 cause all the same instance

    config.set_option('allow_large_jobs', default)
    config.set_option('large_job', default_large)
