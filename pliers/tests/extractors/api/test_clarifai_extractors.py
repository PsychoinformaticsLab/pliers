from os.path import join
from ...utils import get_test_data_path
from pliers import config
from pliers.extractors import (ClarifaiAPIImageExtractor,
                               ClarifaiAPIVideoExtractor)
from pliers.extractors.base import merge_results
from pliers.stimuli import ImageStim, VideoStim
import numpy as np
import pytest

IMAGE_DIR = join(get_test_data_path(), 'image')
VIDEO_DIR = join(get_test_data_path(), 'video')


@pytest.mark.requires_payment
@pytest.mark.skipif("'CLARIFAI_API_KEY' not in os.environ")
def test_clarifai_api_extractor():
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    ext = ClarifaiAPIImageExtractor()
    assert ext.validate_keys()
    result = ext.transform(stim).to_df()
    assert result['apple'][0] > 0.5
    assert result.ix[:, 5][0] > 0.0

    result = ClarifaiAPIImageExtractor(max_concepts=5).transform(stim).to_df()
    assert result.shape == (1, 9)

    result = ClarifaiAPIImageExtractor(
        min_value=0.9).transform(stim).to_df(object_id=False)
    assert all(np.isnan(d) or d > 0.9 for d in result.values[0, 3:])

    concepts = ['cat', 'dog']
    result = ClarifaiAPIImageExtractor(select_concepts=concepts).transform(stim)
    result = result.to_df()
    assert result.shape == (1, 6)
    assert 'cat' in result.columns and 'dog' in result.columns

    url = 'https://via.placeholder.com/350x150'
    stim = ImageStim(url=url)
    result = ClarifaiAPIImageExtractor(max_concepts=5).transform(stim).to_df()
    assert result.shape == (1, 9)
    assert result['graphic'][0] > 0.8

    ext = ClarifaiAPIImageExtractor(api_key='nogood')
    assert not ext.validate_keys()


@pytest.mark.requires_payment
@pytest.mark.skipif("'CLARIFAI_API_KEY' not in os.environ")
def test_clarifai_api_extractor_batch():
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    stim2 = ImageStim(join(IMAGE_DIR, 'obama.jpg'))
    ext = ClarifaiAPIImageExtractor()
    results = ext.transform([stim, stim2])
    results = merge_results(results)
    assert results['ClarifaiAPIImageExtractor#apple'][0] > 0.5 or \
        results['ClarifaiAPIImageExtractor#apple'][1] > 0.5


@pytest.mark.requires_payment
@pytest.mark.skipif("'CLARIFAI_API_KEY' not in os.environ")
def test_clarifai_api_extractor_large():
    default = config.get_option('allow_large_jobs')
    default_large = config.get_option('large_job')
    config.set_option('allow_large_jobs', False)
    config.set_option('large_job', 1)

    ext = ClarifaiAPIImageExtractor()
    images = [ImageStim(join(IMAGE_DIR, 'apple.jpg')),
              ImageStim(join(IMAGE_DIR, 'obama.jpg'))]
    with pytest.raises(ValueError):
        merge_results(ext.transform(images))

    config.set_option('allow_large_jobs', True)
    results = merge_results(ext.transform(images))
    assert 'ClarifaiAPIImageExtractor#apple' in results.columns
    assert results.shape == (2, 49)

    config.set_option('allow_large_jobs', default)
    config.set_option('large_job', default_large)


@pytest.mark.requires_payment
@pytest.mark.skipif("'CLARIFAI_API_KEY' not in os.environ")
def test_clarifai_api_video_extractor():
    stim = VideoStim(join(VIDEO_DIR, 'small.mp4'))
    ext = ClarifaiAPIVideoExtractor()
    assert ext.validate_keys()
    result = ext.transform(stim).to_df()
    # This should actually be 6, in principle, because the clip is < 6 seconds,
    # but the Clarifai API is doing weird things. See comment in
    # ClarifaiAPIVideoExtractor._to_df() for further explanation.
    assert result.shape[1] == 29
    assert result.shape[0] in (6, 7)
    assert result['toy'][0] > 0.5
    assert result['onset'][1] == 1.0
    assert result['duration'][0] == 1.0
    # because of the behavior described above—handle both cases
    assert np.isclose(result['duration'][5], 0.57) or result['duration'][6] == 0
