from unittest import TestCase
from .utils import _get_test_data_path
from featurex.stims import (VideoStim, VideoFrameStim, ComplexTextStim,
                               AudioStim)
from featurex.extractors import Extractor
from featurex.stims import Stim
from featurex.core import Note
from featurex.core import Event
import numpy as np
from os.path import join


class TestStims(TestCase):

    @classmethod
    def setUpClass(self):

        class DummyExtractor(Extractor):

            target = Stim

            def apply(self, stim):
                return Note(stim, self, {'constant': 1})

        class DummyIterableExtractor(Extractor):

            target = Stim

            def apply(self, stim):

                events = []
                time_bins = np.arange(0., stim.duration, 1.)
                for i, tb in enumerate(time_bins):
                    ev = Event(onset=tb, duration=1000)
                    ev.add_note(Note(stim, self, {'second': i}))
                return events

        self.dummy_extractor = DummyExtractor()
        self.dummy_iter_extractor = DummyIterableExtractor()

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
        self.assertEquals(round(stim.duration), 57)
        self.assertEquals(stim.sampling_rate, 11025)
        stim.extract([self.dummy_iter_extractor])

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
