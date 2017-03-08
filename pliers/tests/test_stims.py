from .utils import get_test_data_path
from pliers.stimuli import (VideoStim, VideoFrameStim, ComplexTextStim,
                            AudioStim, ImageStim, CompoundStim,
                            TranscribedAudioCompoundStim,
                            TextStim)
from pliers.stimuli.base import Stim, _get_stim_class
from pliers.extractors import BrightnessExtractor
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.support.download import download_nltk_data
import numpy as np
from os.path import join
import pandas as pd
import pytest


class DummyExtractor(Extractor):

    _input_type = Stim

    def _extract(self, stim):
        return ExtractorResult(np.array([[1]]), stim, self,
                               features=['constant'])


class DummyIterableExtractor(Extractor):

    _input_type = Stim

    def _extract(self, stim):
        time_bins = np.arange(0., stim.duration, 1.)
        return ExtractorResult(np.array([1] * len(time_bins)), stim, self,
                               features=['constant'], onsets=time_bins,
                               durations=[1.] * len(time_bins))


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


def test_video_stim():
    ''' Test VideoStim functionality. '''
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename)
    assert video.fps == 30
    assert video.n_frames in (167, 168)
    assert video.width == 560

    # Test frame iterator
    frames = [f for f in video]
    assert len(frames) == 168
    f1 = frames[100]
    assert isinstance(f1, VideoFrameStim)
    assert isinstance(f1.onset, float)
    f1.data.shape == (320, 560, 3)

    # Test getting of specific frame
    f2 = video.get_frame(index=100)
    assert isinstance(f2, VideoFrameStim)
    assert isinstance(f2.onset, float)
    f2.data.shape == (320, 560, 3)


def test_audio_stim(dummy_iter_extractor):
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'barber.wav'), sampling_rate=11025)
    assert round(stim.duration) == 57
    assert stim.sampling_rate == 11025


def test_audio_formats():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'crowd.mp3'))
    assert round(stim.duration) == 28
    assert stim.sampling_rate == 44100


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
    stim = ComplexTextStim(text=text)
    target = ['To', 'Sherlock', 'Holmes']
    assert [w.text for w in stim.elements[:3]] == target
    assert len(stim.elements) == 231
    stim = ComplexTextStim(text=text, unit='sent')
    # Custom tokenizer
    stim = ComplexTextStim(text=text, tokenizer='(\w+)')
    assert len(stim.elements) == 209


def test_complex_stim_from_srt():
    srtfile = join(get_test_data_path(), 'text', 'wonderful.srt')
    textfile = join(get_test_data_path(), 'text', 'wonderful.txt')
    df = pd.read_csv(textfile, sep='\t')
    target = df["text"].tolist()
    srt_stim = ComplexTextStim(srtfile)
    texts = [sent.text for sent in srt_stim.elements]
    assert texts == target


def test_get_stim():
    assert issubclass(_get_stim_class('video'), VideoStim)
    assert issubclass(_get_stim_class('ComplexTextStim'), ComplexTextStim)
    assert issubclass(_get_stim_class('video_frame'), VideoFrameStim)


def test_compound_stim():
    audio_dir = join(get_test_data_path(), 'audio')
    audio = AudioStim(join(audio_dir, 'crowd.mp3'))
    image1 = ImageStim(join(get_test_data_path(), 'image', 'apple.jpg'))
    image2 = ImageStim(join(get_test_data_path(), 'image', 'obama.jpg'))
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename)
    text = ComplexTextStim(text="The quick brown fox jumped...")
    stim = CompoundStim([audio, image1, image2, video, text])
    assert len(stim.elements) == 5
    assert isinstance(stim.video, VideoStim)
    assert isinstance(stim.complex_text, ComplexTextStim)
    assert isinstance(stim.image, ImageStim)
    with pytest.raises(AttributeError):
        stim.nonexistent_type
    assert stim.video_frame is None

    imgs = stim.get_stim(ImageStim, return_all=True)
    assert len(imgs) == 2
    assert all([isinstance(im, ImageStim) for im in imgs])
    also_imgs = stim.get_stim('image', return_all=True)
    assert imgs == also_imgs


def test_transformations_on_compound_stim():
    image1 = ImageStim(join(get_test_data_path(), 'image', 'apple.jpg'))
    image2 = ImageStim(join(get_test_data_path(), 'image', 'obama.jpg'))
    text = ComplexTextStim(text="The quick brown fox jumped...")
    stim = CompoundStim([image1, image2, text])

    ext = BrightnessExtractor()
    results = ext.transform(stim)
    assert len(results) == 2
    assert np.allclose(results[0].data[0], 0.88784294)


def test_transcribed_audio_stim():
    audio = AudioStim(join(get_test_data_path(), 'audio', "barber_edited.wav"))
    text_file = join(get_test_data_path(), 'text', "wonderful_edited.srt")
    text = ComplexTextStim(text_file)
    stim = TranscribedAudioCompoundStim(audio=audio, text=text)
    assert isinstance(stim.audio, AudioStim)
    assert isinstance(stim.complex_text, ComplexTextStim)


def test_remote_stims():
    url = 'http://www.obamadownloads.com/videos/iran-deal-speech.mp4'
    video = VideoStim(url=url)
    assert video.fps == 12

    url = 'http://www.bobainsworth.com/wav/simpsons/themodyn.wav'
    audio = AudioStim(url=url)
    assert round(audio.duration) == 3

    url = 'https://www.whitehouse.gov/sites/whitehouse.gov/files/images/twitter_cards_potus.jpg'
    image = ImageStim(url=url)
    assert image.data.shape == (240, 240, 3)

    url = 'https://github.com/tyarkoni/pliers/blob/master/README.md'
    text = TextStim(url=url)
    assert len(text.text) > 1
