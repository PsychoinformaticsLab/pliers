from os.path import join
from ...utils import get_test_data_path
from pliers.converters import RevAISpeechAPIConverter
from pliers.stimuli import AudioStim, ComplexTextStim, TextStim
import pytest

AUDIO_DIR = join(get_test_data_path(), 'audio')


@pytest.mark.requires_payment
@pytest.mark.skipif("'REVAI_ACCESS_TOKEN' not in os.environ")
def test_googleAPI_converter():
    stim = AudioStim(join(AUDIO_DIR, 'obama_speech.wav'), onset=4.2)
    conv = RevAISpeechAPIConverter()
    assert conv.validate_keys()
    out_stim = conv.transform(stim)
    assert type(out_stim) == ComplexTextStim
    text = [elem.text for elem in out_stim]
    assert 'years' in text
    assert 'together' in text

    first_word = next(w for w in out_stim)
    assert isinstance(first_word, TextStim)
    assert first_word.duration > 0
    assert first_word.onset > 4.2 and first_word.onset < 8.0

    conv = RevAISpeechAPIConverter(access_token='badtoken')
    assert not conv.validate_keys()
