from os.path import join
from ...utils import get_test_data_path
from pliers import config
from pliers.extractors import ClarifaiAPIExtractor
from pliers.extractors.base import merge_results
from pliers.stimuli import ImageStim, VideoStim
import numpy as np
import pytest


@pytest.mark.skipif("'CLARIFAI_API_KEY' not in os.environ")
def test_clarifai_api_extractor():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
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

    ext = ClarifaiAPIExtractor(api_key='nogood')
    assert not ext.validate_keys()
    with pytest.raises(ValueError):
        ext.transform(stim)


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


@pytest.mark.skipif("'CLARIFAI_API_KEY' not in os.environ")
def test_clarifai_api_extractor_large():
    default = config.get_option('allow_large_jobs')
    config.set_option('allow_large_jobs', False)

    ext = ClarifaiAPIExtractor()
    video = VideoStim(join(get_test_data_path(), 'video', 'small.mp4'))
    with pytest.raises(ValueError):
        merge_results(ext.transform(video))

    images = [ImageStim(join(get_test_data_path(), 'image', 'apple.jpg'))] * 101
    with pytest.raises(ValueError):
        merge_results(ext.transform(images))

    config.set_option('allow_large_jobs', True)
    results = merge_results(ext.transform(images))
    assert 'ClarifaiAPIExtractor#apple' in results.columns
    assert results.shape == (1, 29)  # not 101 cause all the same instance

    config.set_option('allow_large_jobs', default)
