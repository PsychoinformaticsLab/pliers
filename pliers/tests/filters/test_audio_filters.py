from os.path import join
from ..utils import get_test_data_path
from pliers.filters import (AudioTrimmingFilter,
                            TemporalTrimmingFilter,
                            AeneasForcedAlignmentFilter)
from pliers.stimuli import (AudioStim,
                            TranscribedAudioCompoundStim,
                            ComplexTextStim)
import pytest

AUDIO_DIR = join(get_test_data_path(), 'audio')


def test_audio_trimming_filter():
    audio = AudioStim(join(AUDIO_DIR, 'homer.wav'), onset=0.0)
    filt = TemporalTrimmingFilter(start=1.0, end=3.0)
    short_audio = filt.transform(audio)
    assert short_audio.duration == 2.0
    assert short_audio.sampling_rate == audio.sampling_rate

    frame_filt = AudioTrimmingFilter(start=1.5)
    short_audio = frame_filt.transform(audio)
    assert short_audio.onset == 0.0
    assert short_audio.duration == 1.9

    long_filt = AudioTrimmingFilter(start=1.0, end=5.0)
    short_audio = long_filt.transform(audio)
    assert short_audio.duration == audio.duration - 1.0
    assert short_audio.sampling_rate == audio.sampling_rate

    error_filt = AudioTrimmingFilter(start=1.0, end=5.0, validation='strict')
    with pytest.raises(ValueError):
        short_audio = error_filt.transform(audio)


def test_aeneas_alignment_filter():
    aud = AudioStim(join(AUDIO_DIR, 'obama_speech.wav'))
    txt = ComplexTextStim(text='today after two years of negotiations the '
                               'united states together with our international '
                               'partners has achieved')
    stim = TranscribedAudioCompoundStim(aud, txt)
    filt = AeneasForcedAlignmentFilter()
    result = filt.transform(stim).get_stim(ComplexTextStim)

    assert len(result.elements) == 16
    assert result.elements[1].text == 'after'
    assert result.elements[1].onset > 1.5 and result.elements[1].onset < 4.0
    assert (result.elements[6].onset - result.elements[5].onset) > 0.5
    assert result.elements[6].text == 'the'
    assert result.elements[6].onset > 4.0 and result.elements[6].onset < 6.0
    assert (result.elements[9].onset - result.elements[8].onset) > 0.5
    assert (result.elements[14].onset - result.elements[13].onset) > 0.5
