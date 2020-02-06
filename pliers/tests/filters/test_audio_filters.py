from os.path import join
from ..utils import get_test_data_path
from pliers.filters import (AudioTrimmingFilter, 
                            TemporalTrimmingFilter,
                            AudioResamplingFilter)
from pliers.stimuli import AudioStim
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
    

@pytest.mark.parametrize('target_sr', [8000, 22050])
@pytest.mark.parametrize('resample_type', ['kaiser_best', 'kaiser_fast', 'scipy', 'fft', 'polyphase'])
def test_audio_resampling_filter(target_sr, resample_type):

    stim = AudioStim(join(AUDIO_DIR, 'homer.wav'))
    filt = AudioResamplingFilter(target_sr, resample_type)
    resampled = filt.transform(stim)
    
    assert resampled.sampling_rate == target_rs
    assert np.abs(target_sr * stim.duration - resampled.shape[0]) <= 1
