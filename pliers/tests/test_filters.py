from os.path import join
from .utils import get_test_data_path
from pliers.filters import (WordStemmingFilter,
                            FrameSamplingFilter)
from pliers.stimuli import ComplexTextStim, VideoStim, VideoFrameStim
from nltk import stem as nls
import math
import pytest


TEXT_DIR = join(get_test_data_path(), 'text')


def test_word_stemming_filter():
    stim = ComplexTextStim(join(TEXT_DIR, 'sample_text.txt'),
                           columns='to', default_duration=1)

    # With all defaults (porter stemmer)
    filt = WordStemmingFilter()
    assert isinstance(filt.stemmer, nls.PorterStemmer)
    stemmed = filt.transform(stim)
    stems = [s.text for s in stemmed]
    target = ['some', 'sampl', 'text', 'for', 'test', 'annot']
    assert stems == target

    # Try a different stemmer
    filt = WordStemmingFilter(stemmer='snowball', language='english')
    assert isinstance(filt.stemmer, nls.SnowballStemmer)
    stemmed = filt.transform(stim)
    stems = [s.text for s in stemmed]
    assert stems == target

    # Handles StemmerI stemmer
    stemmer = nls.SnowballStemmer(language='english')
    filt = WordStemmingFilter(stemmer=stemmer)
    stemmed = filt.transform(stim)
    stems = [s.text for s in stemmed]
    assert stems == target

    # Fails on invalid values
    with pytest.raises(ValueError):
        filt = WordStemmingFilter(stemmer='nonexistent_stemmer')


def test_frame_sampling_video_converter():
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename, onset=4.2)
    assert video.fps == 30
    assert video.n_frames in (167, 168)
    assert video.width == 560

    # Test frame filters
    conv = FrameSamplingFilter(every=3)
    derived = conv.transform(video)
    assert derived.n_frames == math.ceil(video.n_frames / 3.0)
    first = next(f for f in derived)
    assert type(first) == VideoFrameStim
    assert first.name == 'frame[0]'
    assert first.onset == 4.2
    assert first.duration == 3 * (1 / 30.0)
    second = [f for f in derived][1]
    assert second.onset == 4.3

    # Should refilter from original frames
    conv = FrameSamplingFilter(hertz=15)
    derived = conv.transform(derived)
    assert derived.n_frames == math.ceil(video.n_frames / 6.0)
    first = next(f for f in derived)
    assert type(first) == VideoFrameStim
    assert first.duration == 3 * (1 / 15.0)
    second = [f for f in derived][1]
    assert second.onset == 4.4


def test_derived_video_converter_cv2():
    pytest.importorskip('cv2')
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename)

    conv = FrameSamplingFilter(top_n=5)
    derived = conv.transform(video)
    assert derived.n_frames == 5
    assert type(next(f for f in derived)) == VideoFrameStim
