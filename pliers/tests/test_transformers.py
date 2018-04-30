from pliers.transformers import get_transformer
from pliers.extractors import (STFTAudioExtractor, BrightnessExtractor,
                               merge_results)
from pliers.stimuli.base import TransformationLog
from pliers.stimuli import ImageStim, VideoStim, TextStim
from pliers import config
from os.path import join
from .utils import get_test_data_path, DummyExtractor, DummyBatchExtractor
import numpy as np
import pytest


def test_get_transformer_by_name():
    tda = get_transformer('stFtAudioeXtrActOr', base='extractors')
    assert isinstance(tda, STFTAudioExtractor)

    with pytest.raises(KeyError):
        tda = get_transformer('NotRealExtractor')


def test_transformation_history():
    img = ImageStim(join(get_test_data_path(), 'image', 'apple.jpg'))
    ext = DummyExtractor('giraffe')
    res = ext.transform(img).history
    assert isinstance(res, TransformationLog)
    df = res.to_df()
    assert df.shape == (1, 8)
    assert list(df.columns) == ['source_name', 'source_file', 'source_class',
                          'result_name', 'result_file', 'result_class',
                          'transformer_class', 'transformer_params']
    assert df.iloc[0]['transformer_class'] == 'DummyExtractor'
    assert eval(df.iloc[0]['transformer_params'])['param_A'] == 'giraffe'
    assert str(res) == 'ImageStim->DummyExtractor/ExtractorResult'


def test_transform_with_string_input():
    ext = BrightnessExtractor()
    res = ext.transform(join(get_test_data_path(), 'image', 'apple.jpg'))
    np.testing.assert_almost_equal(res.to_df()['brightness'].values[0], 0.887842942)


def test_parallelization():
    # TODO: test that parallelization actually happened (this will likely
    # require some new logging functionality, or introspection). For now we
    # just make sure the parallelized version produces the same result.
    default = config.get_option('parallelize')
    cache_default = config.get_option('cache_transformers')
    config.set_option('cache_transformers', True)

    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename)
    ext = BrightnessExtractor()

    # With parallelization
    config.set_option('parallelize', True)
    result1 = ext.transform(video)

    # Without parallelization
    config.set_option('parallelize', False)
    result2 = ext.transform(video)

    assert result1 == result2
    config.set_option('parallelize', default)
    config.set_option('cache_transformers', cache_default)


def test_batch_transformer():
    cache_default = config.get_option('cache_transformers')
    config.set_option('cache_transformers', False)

    img1 = ImageStim(join(get_test_data_path(), 'image', 'apple.jpg'))
    img2 = ImageStim(join(get_test_data_path(), 'image', 'button.jpg'))
    img3 = ImageStim(join(get_test_data_path(), 'image', 'obama.jpg'))
    ext = DummyBatchExtractor()
    res = merge_results(ext.transform([img1, img2, img3]))
    assert ext.num_calls == 1
    assert res.shape == (3, 10)
    ext = DummyBatchExtractor(batch_size=1)
    res2 = merge_results(ext.transform([img1, img2, img3]))
    assert ext.num_calls == 3
    assert res.equals(res2)

    config.set_option('cache_transformers', cache_default)


def test_batch_transformer_caching():
    cache_default = config.get_option('cache_transformers')
    config.set_option('cache_transformers', True)

    img1 = ImageStim(join(get_test_data_path(), 'image', 'apple.jpg'))
    ext = DummyBatchExtractor(name='penguin')
    res = ext.transform(img1).to_df(timing=False, object_id=False)
    assert ext.num_calls == 1
    assert res.shape == (1, 1)

    img2 = ImageStim(join(get_test_data_path(), 'image', 'button.jpg'))
    img3 = ImageStim(join(get_test_data_path(), 'image', 'obama.jpg'))
    res2 = ext.transform([img1, img2, img2, img3, img3, img1, img2])
    assert ext.num_calls == 3
    assert len(res2) == 7
    assert res2[0] == res2[5] and res2[1] == res2[2] and res2[3] == res2[4]
    res2 = merge_results(res2)
    assert res2.shape == (3, 10)

    config.set_option('cache_transformers', cache_default)


def test_validation_levels(caplog):
    cache_default = config.get_option('cache_transformers')
    config.set_option('cache_transformers', False)

    ext = BrightnessExtractor()
    stim = TextStim(text='hello world')
    with pytest.raises(TypeError):
        ext.transform(stim)
    res = ext.transform(stim, validation='warn')

    log_message = caplog.records[0].message
    assert log_message == ("Transformers of type BrightnessExtractor can "
                  "only be applied to stimuli of type(s) <class 'pliers"
                  ".stimuli.image.ImageStim'> (not type TextStim), and no "
                  "applicable Converter was found.")
    assert not res

    res = ext.transform(stim, validation='loose')
    assert not res
    stim2 = ImageStim(join(get_test_data_path(), 'image', 'apple.jpg'))
    res = ext.transform([stim, stim2], validation='loose')
    assert len(res) == 1
    assert np.isclose(res[0].to_df()['brightness'][0], 0.88784294, 1e-5)

    config.set_option('cache_transformers', cache_default)


def test_caching():
    cache_default = config.get_option('cache_transformers')
    config.set_option('cache_transformers', True)

    img1 = ImageStim(join(get_test_data_path(), 'image', 'apple.jpg'))
    ext = DummyExtractor()
    res = ext.transform(img1)
    assert ext.num_calls == 1
    res2 = ext.transform(img1)
    assert ext.num_calls == 1
    assert res == res2
    config.set_option('cache_transformers', False)
    res3 = ext.transform(img1)
    assert ext.num_calls == 2
    assert res != res3

    config.set_option('cache_transformers', True)
    ext.num_calls = 0
    res = ext.transform(join(get_test_data_path(), 'image', 'apple.jpg'))
    assert ext.num_calls == 1
    res2 = ext.transform(join(get_test_data_path(), 'image', 'apple.jpg'))
    assert ext.num_calls == 1
    assert res == res2

    config.set_option('cache_transformers', cache_default)

def test_versioning():
    ext = DummyBatchExtractor()
    assert ext.VERSION == '0.1'
    ext = BrightnessExtractor()
    assert ext.VERSION >= '1.0'
