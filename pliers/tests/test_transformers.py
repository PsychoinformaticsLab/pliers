from pliers.transformers import get_transformer
from pliers.extractors.base import Extractor
from pliers.extractors.audio import STFTAudioExtractor
from pliers.tests.utils import get_test_data_path, DummyExtractor
from pliers.stimuli.base import TransformationLog
from pliers.stimuli.image import ImageStim
from os.path import join


def test_get_transformer_by_name():
    tda = get_transformer('stFtAudioeXtrActOr', base='extractors')
    assert isinstance(tda, STFTAudioExtractor)


def test_transformation_history():

    img = ImageStim(join(get_test_data_path(), 'image', 'apple.jpg'))
    ext = DummyExtractor('giraffe')
    res = ext.extract(img).history
    assert isinstance(res, TransformationLog)
    df = res.to_df()
    assert df.shape == (1, 8)
    assert list(df.columns) == ['source_name', 'source_file', 'source_class',
                          'result_name', 'result_file', 'result_class',
                          'transformer_class', 'transformer_params']
    assert df.iloc[0]['transformer_class'] == 'DummyExtractor'
    assert eval(df.iloc[0]['transformer_params'])['param_A'] == 'giraffe'
    assert str(res) == 'ImageStim->DummyExtractor/ExtractorResult'
