from pliers.transformers import get_transformer
from pliers.extractors import (STFTAudioExtractor, BrightnessExtractor, ExtractorResult)
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
    default = config.parallelize

    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename)
    ext = BrightnessExtractor()

    # With parallelization
    config.parallelize = True
    result1 = ext.transform(video)

    # Without parallelization
    config.parallelize = False
    result2 = ext.transform(video)

    assert result1 == result2
    config.parallelize = default


def test_batch_transformer():
    img1 = ImageStim(join(get_test_data_path(), 'image', 'apple.jpg'))
    img2 = ImageStim(join(get_test_data_path(), 'image', 'button.jpg'))
    img3 = ImageStim(join(get_test_data_path(), 'image', 'obama.jpg'))
    ext = DummyBatchExtractor()
    res = ExtractorResult.merge_stims(ext.transform([img1, img2, img3]))
    assert ext.num_calls == 1
    assert res.shape == (3, 8)
    ext = DummyBatchExtractor(batch_size=1)
    res2 = ExtractorResult.merge_stims(ext.transform([img1, img2, img3]))
    assert ext.num_calls == 3
    assert res.equals(res2)


def test_validation_levels(caplog):
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


def test_versioning():
    ext = DummyBatchExtractor()
    assert ext.VERSION == '0.1'
    ext = BrightnessExtractor()
    assert ext.VERSION >= '1.0'
