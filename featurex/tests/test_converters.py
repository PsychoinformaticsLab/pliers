from os.path import join
from .utils import get_test_data_path
from featurex.converters.video import FrameSamplingConverter
from featurex.converters.api import (WitTranscriptionConverter, 
                                        GoogleSpeechAPIConverter,
                                        TesseractAPIConverter)
from featurex.converters.google import GoogleVisionAPITextConverter
from featurex.stimuli.video import VideoStim, VideoFrameStim, DerivedVideoStim
from featurex.stimuli.text import ComplexTextStim
from featurex.stimuli.audio import AudioStim
from featurex.stimuli.image import ImageStim

import numpy as np
import math
import pytest


def test_derived_video_stim():
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename)
    assert video.fps == 30
    assert video.n_frames == 168
    assert video.width == 560

    # Test frame filters
    conv = FrameSamplingConverter(every=3)
    derived = conv.transform(video)
    assert len(derived.elements) == math.ceil(video.n_frames / 3.0)
    assert type(next(f for f in derived)) == VideoFrameStim
    assert next(f for f in derived).duration == 3 * (1 / 30.0)

    # Should refilter from original frames
    conv = FrameSamplingConverter(hertz=15)
    derived = conv.transform(derived)
    assert len(derived.elements) == math.ceil(video.n_frames / 6.0)
    assert type(next(f for f in derived)) == VideoFrameStim
    assert next(f for f in derived).duration == 3 * (1 / 15.0)

    # Test filter history
    assert derived.history.shape == (2, 3)
    assert np.array_equal(derived.history['filter'], ['every', 'hertz'])

def test_derived_video_stim_cv2():
    pytest.importorskip('cv2')
    filename = join(get_test_data_path(), 'video', 'small.mp4')
    video = VideoStim(filename)

    conv = FrameSamplingConverter(top_n=5)
    derived = conv.transform(video)
    assert len(derived.elements) == 5
    assert type(next(f for f in derived)) == VideoFrameStim

@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_witaiAPI_converter():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'homer.wav'))
    conv = WitTranscriptionConverter()
    out_stim = conv.transform(stim)
    assert type(out_stim) == ComplexTextStim
    text = [elem.text for elem in out_stim]
    assert 'thermodynamics' in text or 'obey' in text

@pytest.mark.skipif("'GOOGLE_API_KEY' not in os.environ")
def test_googleAPI_converter():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'homer.wav'))
    conv = GoogleSpeechAPIConverter()
    out_stim = conv.transform(stim)
    assert type(out_stim) == ComplexTextStim
    text = [elem.text for elem in out_stim]
    assert 'thermodynamics' in text or 'obey' in text

def test_tesseract_converter():
    pytest.importorskip('pytesseract')
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'button.jpg'))
    conv = TesseractAPIConverter()
    text = conv.transform(stim).text
    assert text == 'Exit'

@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_text_converter():
    conv = GoogleVisionAPITextConverter(num_retries=5)
    filename = join(get_test_data_path(), 'image', 'button.jpg')
    stim = ImageStim(filename)
    text = conv.transform(stim).text
    assert 'Exit' in text
