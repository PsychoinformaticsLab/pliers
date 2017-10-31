from os.path import join
from ..utils import get_test_data_path
from pliers.converters import GoogleVisionAPITextConverter
from pliers.stimuli import ImageStim
import pytest

IMAGE_DIR = join(get_test_data_path(), 'image')


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_text_converter():
    conv = GoogleVisionAPITextConverter(num_retries=5)
    filename = join(IMAGE_DIR, 'button.jpg')
    stim = ImageStim(filename)
    text = conv.transform(stim).text
    assert 'Exit' in text

    conv = GoogleVisionAPITextConverter(handle_annotations='concatenate')
    text = conv.transform(stim).text
    assert 'Exit' in text
