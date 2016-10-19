from os.path import join
from .utils import get_test_data_path
from featurex.converters.api import WitTranscriptionConverter, GoogleSpeechAPIConverter, IBMSpeechAPIConverter
from featurex.stimuli.text import DynamicTextStim, ComplexTextStim
from featurex.stimuli.audio import AudioStim
import pytest


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_witaiAPI_converter():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'homer.wav'))
    conv = WitTranscriptionConverter()
    out_stim = conv.transform(stim)
    assert type(out_stim) == ComplexTextStim
    text = [elem.text for elem in out_stim]
    assert 'thermodynamics' in text or 'obey' in text

@pytest.mark.skipif("'GOOGLE_API_KEY' not in os.environ")
def test_googleAPI_converter():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'homer.wav'))
    conv = GoogleSpeechAPIConverter()
    out_stim = conv.transform(stim)
    assert type(out_stim) == ComplexTextStim
    text = [elem.text for elem in out_stim]
    assert 'thermodynamics' in text or 'obey' in text

@pytest.mark.skipif("'IBM_USERNAME' not in os.environ or "
    "'IBM_PASSWORD' not in os.environ")
def test_ibmAPI_converter():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'homer.wav'))
    conv = IBMSpeechAPIConverter()
    out_stim = conv.transform(stim)
    assert type(out_stim) == ComplexTextStim
    first_word = next(w for w in out_stim)
    assert type(first_word) == DynamicTextStim
    assert first_word.duration > 0
    assert first_word.onset != None

    full_text = [elem.text for elem in out_stim]
    assert 'thermodynamics' in full_text or 'obey' in full_text