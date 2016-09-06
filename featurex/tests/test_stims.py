from unittest import TestCase
from .utils import _get_test_data_path
from featurex.stimuli.video import VideoStim, VideoFrameStim
from featurex.stimuli.text import ComplexTextStim
from featurex.stimuli.audio import AudioStim
from featurex.stimuli.image import ImageStim
from featurex.extractors import Extractor
from featurex.stimuli import Stim
from featurex.core import Value
from featurex.core import Event
from featurex.support.download import download_nltk_data
import numpy as np
from os.path import join


class TestStims(TestCase):

    @classmethod
    def setUpClass(self):

        download_nltk_data()

        class DummyExtractor(Extractor):

            target = Stim

            def apply(self, stim):
                return Value(stim, self, {'constant': 1})

        class DummyIterableExtractor(Extractor):

            target = Stim

            def apply(self, stim):

                events = []
                time_bins = np.arange(0., stim.duration, 1.)
                for i, tb in enumerate(time_bins):
                    ev = Event(onset=tb, duration=1000)
                    ev.add_value(Value(stim, self, {'second': i}))
                return events

        self.dummy_extractor = DummyExtractor()
        self.dummy_iter_extractor = DummyIterableExtractor()

    def test_image_stim(self):
        filename = join(_get_test_data_path(), 'image', 'apple.jpg')
        stim = ImageStim(filename)
        assert stim.data.shape == (288, 420, 3)

    def test_video_stim(self):
        ''' Test VideoStim functionality. '''
        filename = join(_get_test_data_path(), 'video', 'small.mp4')
        video = VideoStim(filename)
        self.assertEquals(video.fps, 30)
        self.assertEquals(video.n_frames, 168)
        self.assertEquals(video.width, 560)

        # Test frame iterator
        frames = [f for f in video]
        self.assertEquals(len(frames), 168)
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

    def test_complex_stim_from_text(self):
        textfile = join(_get_test_data_path(), 'text', 'scandal.txt')
        text = open(textfile).read().strip()
        stim = ComplexTextStim.from_text(text)
        target = ['To', 'Sherlock', 'Holmes']
        self.assertEquals([w.text for w in stim.elements[:3]], target)
        self.assertEquals(len(stim.elements), 231)
        stim = ComplexTextStim.from_text(text, unit='sent')
        # Custom tokenizer
        stim = ComplexTextStim.from_text(text, tokenizer='(\w+)')
        self.assertEquals(len(stim.elements), 209)
