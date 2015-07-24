from unittest import TestCase
from utils import _get_test_data_path
from annotations.stims import (VideoStim, VideoFrameStim, ComplexTextStim,
                               AudioStim)
from os.path import join


class TestStims(TestCase):

    def test_video_stim(self):
        ''' Test VideoStim functionality. '''
        filename = join(_get_test_data_path(), 'video', 'small.mp4')
        video = VideoStim(filename)
        self.assertEquals(video.fps, 30)
        self.assertEquals(video.n_frames, 166)
        self.assertEquals(video.width, 560)

        # Test frame iterator
        frames = [f for f in video]
        self.assertEquals(len(frames), 166)
        f = frames[100]
        self.assertIsInstance(f, VideoFrameStim)
        self.assertIsInstance(f.onset, float)
        self.assertEquals(f.data.shape, (320, 560, 3))

    def test_audio_stim(self):
        audio_dir = join(_get_test_data_path(), 'audio')
        stim = AudioStim(join(audio_dir, 'barber.wav'))
        self.assertEquals(stim.sampling_rate, 11025)

    def test_complex_text_stim(self):
        text_dir = join(_get_test_data_path(), 'text')
        stim = ComplexTextStim(join(text_dir, 'complex_stim_no_header.txt'),
                               columns='ot', default_duration=0.2)
        self.assertEquals(len(stim.elements), 4)
        self.assertEquals(stim.elements[2].onset, 34)
        self.assertEquals(stim.elements[2].duration, 0.2)
        stim = ComplexTextStim(join(text_dir, 'complex_stim_with_header.txt'))
        self.assertEquals(len(stim.elements), 4)
        self.assertEquals(stim.elements[2].duration, 0.1)
