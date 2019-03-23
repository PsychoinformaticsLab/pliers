from .utils import get_test_data_path
from pliers.stimuli import (VideoStim, VideoFrameStim, ComplexTextStim,
                            AudioStim, ImageStim, CompoundStim,
                            TranscribedAudioCompoundStim,
                            TextStim,
                            TweetStimFactory,
                            TweetStim)
from pliers.stimuli.base import Stim, _get_stim_class
from pliers.extractors import (BrightnessExtractor, LengthExtractor,
                               ComplexTextExtractor)
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.support.download import download_nltk_data
import numpy as np
from os.path import join, exists
import pandas as pd
import pytest
import tempfile
import os
import base64


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


def test_image_stim_bytestring():
    path = join(get_test_data_path(), 'image', 'apple.jpg')
    img = ImageStim(path)
    assert img._bytestring is None
    bs = img.get_bytestring()
    assert isinstance(bs, str)
    assert img._bytestring is not None
    raw = bs.encode()
    with open(path, 'rb') as f:
        assert raw == base64.b64encode(f.read())


def test_complex_text_hash():
    stims = [ComplexTextStim(text='yeah'), ComplexTextStim(text='buddy')]
    ext = ComplexTextExtractor()
    res = ext.transform(stims)

    assert res[0]._data != res[1]._data


def test_video_stim():
    ''' Test VideoStim functionality. '''
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename, onset=4.2)
    assert video.fps == 30
    assert video.n_frames == 168
    assert video.width == 560
    assert video.duration == 5.57

    # Test frame iterator
    frames = [f for f in video]
    assert len(frames) == 168
    f1 = frames[100]
    assert isinstance(f1, VideoFrameStim)
    assert isinstance(f1.onset, float)
    assert np.isclose(f1.duration, 1 / 30.0, 1e-5)
    f1.data.shape == (320, 560, 3)

    # Test getting of specific frame
    f2 = video.get_frame(index=100)
    assert isinstance(f2, VideoFrameStim)
    assert isinstance(f2.onset, float)
    assert f2.onset > 7.5
    f2.data.shape == (320, 560, 3)
    f2_copy = video.get_frame(onset=3.33334)
    assert isinstance(f2, VideoFrameStim)
    assert isinstance(f2.onset, float)
    assert f2.onset > 7.5
    assert np.array_equal(f2.data, f2_copy.data)

    # Try another video
    filename = join(get_test_data_path(), 'video', 'obama_speech.mp4')
    video = VideoStim(filename)
    assert video.fps == 12
    assert video.n_frames == 105
    assert video.width == 320
    assert video.duration == 8.71
    f3 = video.get_frame(index=104)
    assert isinstance(f3, VideoFrameStim)
    assert isinstance(f3.onset, float)
    assert f3.duration > 0.0
    assert f3.data.shape == (240, 320, 3)


def test_video_stim_bytestring():
    path = join(get_test_data_path(), 'video', 'small.mp4')
    vid = VideoStim(path)
    assert vid._bytestring is None
    bs = vid.get_bytestring()
    assert isinstance(bs, str)
    assert vid._bytestring is not None
    raw = bs.encode()
    with open(path, 'rb') as f:
        assert raw == base64.b64encode(f.read())


def test_video_frame_stim():
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename, onset=4.2)
    frame = VideoFrameStim(video, 42)
    assert frame.onset == (5.6)
    assert np.array_equal(frame.data, video.get_frame(index=42).data)
    assert frame.name == 'frame[42]'


def test_audio_stim():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'barber.wav'))
    assert round(stim.duration) == 57
    assert stim.sampling_rate == 11025

    stim = AudioStim(join(audio_dir, 'homer.wav'))
    assert round(stim.duration) == 3
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
    stim = ComplexTextStim(join(text_dir, 'complex_stim_no_header.txt'),
                           columns='ot', default_duration=0.2, onset=4.2)
    assert stim.elements[2].onset == 38.2
    assert stim.elements[1].onset == 24.2
    stim = ComplexTextStim(join(text_dir, 'complex_stim_with_header.txt'))
    assert len(stim.elements) == 4
    assert stim.elements[2].duration == 0.1

    assert stim._to_sec((1.0, 42, 3, 0)) == 6123
    assert stim._to_tup(6123) == (1.0, 42, 3, 0)


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

    # Test iteration
    len([e for e in stim]) == 5

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
    assert np.allclose(results[0]._data[0], 0.88784294)


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


def test_get_filename():
    url = 'http://www.bobainsworth.com/wav/simpsons/themodyn.wav'
    audio = AudioStim(url=url)
    with audio.get_filename() as filename:
        assert exists(filename)
    assert not exists(filename)

    url = 'https://via.placeholder.com/350x150'
    image = ImageStim(url=url)
    with image.get_filename() as filename:
        assert exists(filename)
    assert not exists(filename)


def test_save():
    cts_file = join(get_test_data_path(), 'text', 'complex_stim_no_header.txt')
    complextext_stim = ComplexTextStim(cts_file, columns='ot',
                                       default_duration=0.2)
    text_stim = TextStim(text='hello')
    audio_stim = AudioStim(join(get_test_data_path(), 'audio', 'crowd.mp3'))
    image_stim = ImageStim(join(get_test_data_path(), 'image', 'apple.jpg'))
    # Video gives travis problems
    stims = [complextext_stim, text_stim, audio_stim, image_stim]
    for s in stims:
        path = tempfile.mktemp() + s._default_file_extension
        s.save(path)
        assert exists(path)
        os.remove(path)


@pytest.mark.skipif("'TWITTER_ACCESS_TOKEN_KEY' not in os.environ")
def test_twitter():
    # Test stim creation
    pytest.importorskip('twitter')
    factory = TweetStimFactory()
    status_id = 821442726461931521
    pliers_tweet = factory.get_status(status_id)
    assert isinstance(pliers_tweet, TweetStim)
    assert isinstance(pliers_tweet, CompoundStim)
    assert len(pliers_tweet.elements) == 1

    status_id = 884392294014746624
    ut_tweet = factory.get_status(status_id)
    assert len(ut_tweet.elements) == 2

    # Test extraction
    ext = LengthExtractor()
    res = ext.transform(pliers_tweet)[0].to_df()
    assert res['text_length'][0] == 104

    # Test image extraction
    ext = BrightnessExtractor()
    res = ext.transform(ut_tweet)[0].to_df()
    brightness = res['brightness'][0]
    assert np.isclose(brightness, 0.54057, 1e-5)
