from os.path import join, splitext
from .utils import get_test_data_path
from pliers.converters import (get_converter, FrameSamplingConverter,
                             VideoToAudioConverter, VideoToTextConverter,
                             TesseractConverter, WitTranscriptionConverter,
                             GoogleSpeechAPIConverter, IBMSpeechAPIConverter,
                             GoogleVisionAPITextConverter, ComplexTextIterator)
from pliers.converters.image import ImageToTextConverter
from pliers.stimuli import (VideoStim, VideoFrameStim, DerivedVideoStim,
                            TextStim, ComplexTextStim, AudioStim, ImageStim)
from pliers import config
import numpy as np
import math
import pytest
import os
import time


def test_video_to_audio_converter():
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename)
    conv = VideoToAudioConverter()
    audio = conv.transform(video)
    assert audio.name == 'small.wav'
    assert audio.history.source_class == 'VideoStim'
    assert audio.history.source_file == filename
    assert splitext(video.filename)[0] == splitext(audio.filename)[0]
    assert np.isclose(video.duration, audio.duration, 1e-2)


def test_derived_video_converter():
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename)
    assert video.fps == 30
    assert video.n_frames in (167, 168)
    assert video.width == 560

    # Test frame filters
    conv = FrameSamplingConverter(every=3)
    derived = conv.transform(video)
    assert len(derived._frames) == math.ceil(video.n_frames / 3.0)
    first = next(f for f in derived)
    assert type(first) == VideoFrameStim
    assert first.name == 'frame[0]'
    assert first.duration == 3 * (1 / 30.0)

    # Should refilter from original frames
    conv = FrameSamplingConverter(hertz=15)
    derived = conv.transform(derived)
    assert len(derived._frames) == math.ceil(video.n_frames / 6.0)
    first = next(f for f in derived)
    assert type(first) == VideoFrameStim
    assert first.duration == 3 * (1 / 15.0)


def test_derived_video_converter_cv2():
    pytest.importorskip('cv2')
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename)

    conv = FrameSamplingConverter(top_n=5)
    derived = conv.transform(video)
    assert len(derived._frames) == 5
    assert type(next(f for f in derived)) == VideoFrameStim


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_witaiAPI_converter():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'homer.wav'))
    conv = WitTranscriptionConverter()
    out_stim = conv.transform(stim)
    assert type(out_stim) == ComplexTextStim
    first_word = next(w for w in out_stim)
    assert type(first_word) == TextStim
    #assert '_' in first_word.name
    text = [elem.text for elem in out_stim]
    assert 'thermodynamics' in text or 'obey' in text


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_googleAPI_converter():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'homer.wav'))
    conv = GoogleSpeechAPIConverter()
    out_stim = conv.transform(stim)
    assert type(out_stim) == ComplexTextStim
    text = [elem.text for elem in out_stim]
    assert 'thermodynamics' in text or 'obey' in text


@pytest.mark.skipif("'IBM_USERNAME' not in os.environ or "
    "'IBM_PASSWORD' not in os.environ")
def test_ibmAPI_converter():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'homer.wav'))
    conv = IBMSpeechAPIConverter()
    out_stim = conv.transform(stim)
    assert isinstance(out_stim, ComplexTextStim)
    first_word = next(w for w in out_stim)
    assert isinstance(first_word, TextStim)
    assert first_word.duration > 0
    assert first_word.onset != None

    full_text = [elem.text for elem in out_stim]
    assert 'thermodynamics' in full_text or 'obey' in full_text


def test_tesseract_converter():
    pytest.importorskip('pytesseract')
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'button.jpg'))
    conv = TesseractConverter()
    out_stim = conv.transform(stim)
    assert out_stim.name == 'text[Exit]'
    assert out_stim.history.source_class == 'ImageStim'
    assert out_stim.history.source_name == 'button.jpg'


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_text_converter():
    conv = GoogleVisionAPITextConverter(num_retries=5)
    filename = join(get_test_data_path(), 'image', 'button.jpg')
    stim = ImageStim(filename)
    text = conv.transform(stim).text
    assert 'Exit' in text

    conv = GoogleVisionAPITextConverter(handle_annotations='concatenate')
    text = conv.transform(stim).text
    assert 'Exit' in text


def test_get_converter():
    conv = get_converter(ImageStim, TextStim)
    assert isinstance(conv, ImageToTextConverter)
    conv = get_converter(TextStim, ImageStim)
    assert conv is None


# def test_converter_memoization():

#     cache_value = config.cache_converters
#     config.cache_converters = True

#     filename = join(get_test_data_path(), 'video', 'small.mp4')
#     video = VideoStim(filename)
#     conv = VideoToAudioConverter()

#     def convert(stim):
#         start_time = time.time()
#         stim = conv.transform(stim)
#         return time.time() - start_time

#     # Time taken first time through
#     memory.clear()

#     convert_time = convert(video)
#     cache_time = convert(video)

#     # TODO: implement saner checking than this
#     # Converting should be at least twice as slow as retrieving from cache
#     assert convert_time >= cache_time * 2

#     # After clearing the cache, checks should fail
#     memory.clear()
#     cache_time = convert(video)
#     assert convert_time <= cache_time * 2

#     # When cach is disabled, check should also fail
#     config.cache_converters = False
#     conv = VideoToAudioConverter()
#     cache_time = convert(video)
#     assert convert_time <= cache_time * 2

#     config.cache_converters = cache_value


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
    assert str(stim.history) == 'VideoStim->VideoToAudioConverter/AudioStim->WitTranscriptionConverter/ComplexTextStim'


def test_stim_iteration_converter():
    textfile = join(get_test_data_path(), 'text', 'scandal.txt')
    stim = ComplexTextStim(text=open(textfile).read().strip())
    words = ComplexTextIterator().transform(stim)
    assert len(words) == 231
    assert isinstance(words[1], TextStim)
    assert words[1].text == 'Sherlock'
    assert str(words[1].history) == 'ComplexTextStim->ComplexTextIterator/TextStim'
