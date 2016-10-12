from .utils import get_test_data_path
from featurex.stimuli.video import VideoStim, VideoFrameStim, DerivedVideoStim
from featurex.converters.video import FrameSamplingConverter
import numpy as np
from os.path import join
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

    conv = FrameSamplingConverter(num_frames=5)
    derived = conv.transform(video)
    assert len(derived.elements) == 5
    assert type(next(f for f in derived)) == VideoFrameStim