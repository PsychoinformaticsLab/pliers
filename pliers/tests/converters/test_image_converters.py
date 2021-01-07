from os.path import join

import pytest

from ..utils import get_test_data_path
from pliers.converters import TesseractConverter
from pliers.stimuli import ImageStim

IMAGE_DIR = join(get_test_data_path(), 'image')


def test_tesseract_converter():
    pytest.importorskip('pytesseract')
    stim = ImageStim(join(IMAGE_DIR, 'button.jpg'), onset=4.2)
    conv = TesseractConverter()
    out_stim = conv.transform(stim)
    assert 'Exit' in out_stim.name
    assert out_stim.history.source_class == 'ImageStim'
    assert out_stim.history.source_name == 'button.jpg'
    assert out_stim.onset == 4.2
