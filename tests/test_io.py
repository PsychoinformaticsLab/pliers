from unittest import TestCase
from .utils import _get_test_data_path
from annotations.io import load
from os.path import join
from six import string_types


class TestCore(TestCase):

    def test_magic_loader(self):
        text_file = join(_get_test_data_path(), 'text', 'sample_text.txt')
        audio_file = join(_get_test_data_path(), 'audio', 'barber.wav')
        video_file = join(_get_test_data_path(), 'video', 'small.mp4')
        stim_files = [text_file, audio_file, video_file]
        stims = load(stim_files)
        assert len(stims) == 3
        assert round(stims[1].duration) == 57
        assert isinstance(stims[0].text, string_types)
        assert stims[2].width == 560
