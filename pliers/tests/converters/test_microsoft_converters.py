from os.path import join
from ..utils import get_test_data_path
from pliers.converters import MicrosoftAPITextConverter
from pliers.stimuli import ImageStim
import pytest

IMAGE_DIR = join(get_test_data_path(), 'image')


@pytest.mark.skipif("'MICROSOFT_VISION_SUBSCRIPTION_KEY' not in os.environ")
def test_microsoft_vision_api_text_converter():
    conv = MicrosoftAPITextConverter()
    img = ImageStim(join(IMAGE_DIR, 'button.jpg'))
    text = conv.transform(img).text
    assert 'Exit' in text

    conv = MicrosoftAPITextConverter()
    img = ImageStim(join(IMAGE_DIR, 'CC0', '28010844841_c5b81cb9cc_z.jpg'))
    text = conv.transform(img).text
    assert 'Santander' in text
