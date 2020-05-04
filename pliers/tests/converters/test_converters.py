from os.path import join

import pytest

from ..utils import get_test_data_path
from pliers.converters import (get_converter,
                               VideoToAudioConverter,
                               VideoToTextConverter,
                               WitTranscriptionConverter,
                               ComplexTextIterator,
                               ExtractorResultToSeriesConverter)
from pliers.converters.image import ImageToTextConverter
from pliers.stimuli import (VideoStim, TextStim, SeriesStim,
                            ComplexTextStim, ImageStim)
from pliers.extractors import ExtractorResult


def test_get_converter():
    conv = get_converter(ImageStim, TextStim)
    assert isinstance(conv, ImageToTextConverter)
    conv = get_converter(TextStim, ImageStim)
    assert conv is None


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_multistep_converter():
    conv = VideoToTextConverter()
    filename = join(get_test_data_path(), 'video', 'obama_speech.mp4')
    stim = VideoStim(filename)
    text = conv.transform(stim)
    assert isinstance(text, ComplexTextStim)
    first_word = next(w for w in text)
    assert type(first_word) == TextStim


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_stim_history_tracking():
    video = VideoStim(join(get_test_data_path(), 'video', 'obama_speech.mp4'))
    assert video.history is None
    conv = VideoToAudioConverter()
    stim = conv.transform(video)
    assert str(stim.history) == 'VideoStim->VideoToAudioConverter/AudioStim'
    conv = WitTranscriptionConverter()
    stim = conv.transform(stim)
    assert str(
        stim.history) == 'VideoStim->VideoToAudioConverter/AudioStim->WitTranscriptionConverter/ComplexTextStim'


def test_stim_iteration_converter():
    textfile = join(get_test_data_path(), 'text', 'scandal.txt')
    stim = ComplexTextStim(text=open(textfile).read().strip())
    words = ComplexTextIterator().transform(stim)
    assert len(words) == 231
    assert isinstance(words[1], TextStim)
    assert words[1].text == 'Sherlock'
    assert str(
        words[1].history) == 'ComplexTextStim->ComplexTextIterator/TextStim'


def test_extractor_result_to_series_converter():
    data = [[2, 4], [1, 7], [6, 6], [8, 2]]
    result = ExtractorResult(data, None, None, features=['a', 'b'],
                             onsets=[2, 4, 6, 8])
    stims = ExtractorResultToSeriesConverter().transform(result)
    assert len(stims) == 4
    stim = stims[2]
    assert isinstance(stim, SeriesStim)
    assert stim.data.shape == (2,)
    assert list(stim.data) == [6, 6]
    assert stim.onset == 6
    assert stim.duration is None
    assert stim.order == 2
