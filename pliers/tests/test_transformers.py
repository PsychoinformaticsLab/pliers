from pliers.transformers import get_transformer
from pliers.extractors import Extractor
from pliers.extractors.audio import STFTAudioExtractor


def test_get_transformer_by_name():
    tda = get_transformer('stFtAudioeXtrActOr', base=Extractor)
    assert isinstance(tda, STFTAudioExtractor)




