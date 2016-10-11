from os.path import join
from .utils import get_test_data_path
from featurex.converters.api import WitTranscriptionConverter, GoogleSpeechAPIConverter
from featurex.stimuli.text import TextStim, ComplexTextStim
from featurex.stimuli.video import ImageStim, VideoStim
from featurex.stimuli.audio import AudioStim, TranscribedAudioStim
import pytest


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_witaiAPI_converter():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'homer.wav'))
    conv = WitTranscriptionConverter()
    out_stim = conv.transform(stim)
    assert type(out_stim) == TextStim
    assert 'laws of thermodynamics' in out_stim.text or 'we obey' in out_stim.text

@pytest.mark.skipif("'GOOGLE_API_KEY' not in os.environ")
def test_googleAPI_converter():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'homer.wav'))
    conv = GoogleSpeechAPIConverter()
    out_stim = conv.transform(stim)
    assert type(out_stim) == TextStim
    assert 'laws of thermodynamics' in out_stim.text or 'we obey' in out_stim.text
