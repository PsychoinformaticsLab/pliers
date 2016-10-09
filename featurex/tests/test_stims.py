from .utils import get_test_data_path
from featurex.stimuli.video import VideoStim, VideoFrameStim, DerivedVideoStim
from featurex.stimuli.text import ComplexTextStim
from featurex.stimuli.audio import AudioStim
from featurex.stimuli.image import ImageStim
from featurex.extractors import Extractor
from featurex.stimuli import Stim
from featurex.core import Value, Event, Timeline
from featurex.support.download import download_nltk_data
import numpy as np
from os.path import join
import pandas as pd
import math
import pytest


class DummyExtractor(Extractor):

    target = Stim

    def _extract(self, stim):
        return Value(stim, self, {'constant': 1})


class DummyIterableExtractor(Extractor):

    target = Stim

    def _extract(self, stim):

        events = []
        time_bins = np.arange(0., stim.duration, 1.)
        for i, tb in enumerate(time_bins):
            ev = Event(onset=tb, duration=1000)
            ev.add_value(Value(stim, self, {'second': i}))
        return events


@pytest.fixture(scope='module')
def get_nltk():
    download_nltk_data()


@pytest.fixture(scope='module')
def dummy_extractor():
    return DummyExtractor()


@pytest.fixture(scope='module')
def dummy_iter_extractor():
    return DummyIterableExtractor()


def test_image_stim(dummy_iter_extractor):
    filename = join(get_test_data_path(), 'image', 'apple.jpg')
    stim = ImageStim(filename)
    assert stim.data.shape == (288, 420, 3)
    values = stim.extract([dummy_iter_extractor])
    assert isinstance(values, Value)

def test_video_stim():
    ''' Test VideoStim functionality. '''
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename)
    assert video.fps == 30
    assert video.n_frames == 168
    assert video.width == 560

    # Test frame iterator
    frames = [f for f in video]
    assert len(frames) == 168
    f = frames[100]
    assert isinstance(f, VideoFrameStim)
    assert isinstance(f.onset, float)
    f.data.shape == (320, 560, 3)

def test_derived_video_stim():
    ''' Test DerivedVideoStim functionality. '''
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = DerivedVideoStim(filename)
    assert video.fps == 30
    assert video.n_frames == 168
    assert video.width == 560

    # Test frame filters
    video.filter(every=3)
    assert len(video.elements) == math.ceil(video.n_frames / 3.0)
    assert type(next(f for f in video)) == VideoFrameStim
    assert next(f for f in video).duration == 3 * (1 / 30.0)

    # Should refilter from original frames
    video.filter(hertz=15)
    assert len(video.elements) == math.ceil(video.n_frames / 2.0)
    assert type(next(f for f in video)) == VideoFrameStim
    assert next(f for f in video).duration == 1 * (1 / 15.0)

    # Test filter history
    assert video.history.shape == (3, 3)
    assert np.array_equal(video.history['filter'], ['None', 'every', 'hertz'])

def test_derived_video_stim_cv2():
    pytest.importorskip('cv2')
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = DerivedVideoStim(filename)

    video.filter(num_frames=5)
    assert len(video.elements) == 5
    assert type(next(f for f in video)) == VideoFrameStim

def test_audio_stim(dummy_iter_extractor):
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'barber.wav'))
    assert round(stim.duration) == 57
    assert stim.sampling_rate == 11025
    tl = stim.extract([dummy_iter_extractor])
    assert isinstance(tl, Timeline)

def test_complex_text_stim():
    text_dir = join(get_test_data_path(), 'text')
    stim = ComplexTextStim(join(text_dir, 'complex_stim_no_header.txt'),
                           columns='ot', default_duration=0.2)
    assert len(stim.elements) == 4
    assert stim.elements[2].onset == 34
    assert stim.elements[2].duration == 0.2
    stim = ComplexTextStim(join(text_dir, 'complex_stim_with_header.txt'))
    assert len(stim.elements) == 4
    assert stim.elements[2].duration == 0.1

def test_complex_stim_from_text():
    textfile = join(get_test_data_path(), 'text', 'scandal.txt')
    text = open(textfile).read().strip()
    stim = ComplexTextStim.from_text(text)
    target = ['To', 'Sherlock', 'Holmes']
    assert [w.text for w in stim.elements[:3]] == target
    assert len(stim.elements) == 231
    stim = ComplexTextStim.from_text(text, unit='sent')
    # Custom tokenizer
    stim = ComplexTextStim.from_text(text, tokenizer='(\w+)')
    assert len(stim.elements) == 209
    
def test_complex_stim_from_srt():
    srtfile = join(get_test_data_path(), 'text', 'wonderful.srt')
    textfile = join(get_test_data_path(), 'text', 'wonderful.txt')
    df = pd.read_csv(textfile, sep='\t')
    target = df["text"].tolist()
    srt_stim = ComplexTextStim(srtfile)
    texts = [sent.text for sent in srt_stim.elements]
    assert texts == target
