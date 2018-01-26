from os.path import join
from ..utils import get_test_data_path
from pliers.filters import AudioTrimmingFilter
from pliers.stimuli import AudioStim

AUDIO_DIR = join(get_test_data_path(), 'audio')


def test_audio_trimming_filter():
    audio = AudioStim(join(AUDIO_DIR, 'homer.wav'), onset=0.0)
    filt = AudioTrimmingFilter(start=1.0, end=3.0)
    short_video = filt.transform(audio)
    assert short_video.duration == 2.0
    assert short_video.sampling_rate == audio.sampling_rate

    frame_filt = AudioTrimmingFilter(start=1.5)
    short_video = frame_filt.transform(audio)
    assert short_video.onset == 0.0
    assert short_video.duration == 1.9
