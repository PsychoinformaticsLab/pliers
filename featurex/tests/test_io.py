from .utils import _get_test_data_path
from featurex.stimuli import load_stims
from os.path import join
from six import string_types


def test_magic_loader():
    text_file = join(_get_test_data_path(), 'text', 'sample_text.txt')
    audio_file = join(_get_test_data_path(), 'audio', 'barber.wav')
    video_file = join(_get_test_data_path(), 'video', 'small.mp4')
    stim_files = [text_file, audio_file, video_file]
    stims = load_stims(stim_files)
    assert len(stims) == 3
    assert round(stims[1].duration) == 57
    assert isinstance(stims[0].text, string_types)
    assert stims[2].width == 560
