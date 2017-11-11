from os.path import join
from ..utils import get_test_data_path
from pliers.converters import (GoogleVisionAPITextConverter,
                               GoogleSpeechAPIConverter)
from pliers.stimuli import ImageStim, AudioStim, ComplexTextStim
import pytest

IMAGE_DIR = join(get_test_data_path(), 'image')
AUDIO_DIR = join(get_test_data_path(), 'audio')


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_googleAPI_converter():
    stim = AudioStim(join(AUDIO_DIR, 'homer.wav'))
    conv = GoogleSpeechAPIConverter()
    out_stim = conv.transform(stim)
    assert type(out_stim) == ComplexTextStim
    text = [elem.text for elem in out_stim]
    assert 'thermodynamics' in text or 'obey' in text


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
