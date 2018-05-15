from os.path import join
from ..utils import get_test_data_path
from pliers.filters import (FrameSamplingFilter,
                            VideoTrimmingFilter,
                            TemporalTrimmingFilter)
from pliers.stimuli import VideoStim, VideoFrameStim
import pytest
import math

VIDEO_DIR = join(get_test_data_path(), 'video')


def test_frame_sampling_video_filter():
    filename = join(VIDEO_DIR, 'small.mp4')
    video = VideoStim(filename, onset=4.2)
    assert video.fps == 30
    assert video.n_frames in (167, 168)
    assert video.width == 560

    # Test frame filters
    conv = FrameSamplingFilter(every=3)
    derived = conv.transform(video)
    assert derived.n_frames == math.ceil(video.n_frames / 3.0)
    assert derived.duration == video.duration
    first = next(f for f in derived)
    assert type(first) == VideoFrameStim
    assert first.name == 'frame[0]'
    assert first.onset == 4.2
    assert first.duration == 3 * (1 / 30.0)
    second = [f for f in derived][1]
    assert second.onset == 4.3
    with pytest.raises(TypeError):
        derived.get_frame(onset=1.0)

    # Commented out because no longer allowing sampling filter chaining
    # conv = FrameSamplingFilter(hertz=15)
    # derived = conv.transform(derived)
    # assert derived.n_frames == math.ceil(video.n_frames / 6.0)
    # first = next(f for f in derived)
    # assert type(first) == VideoFrameStim
    # assert first.duration == 3 * (1 / 15.0)
    # second = [f for f in derived][1]
    # assert second.onset == 4.4

    with pytest.raises(TypeError):
        conv.transform(derived)


def test_frame_sampling_video_filter2():
    filename = join(VIDEO_DIR, 'obama_speech.mp4')
    video = VideoStim(filename, onset=4.2)
    assert video.fps == 12
    assert video.n_frames == 105

    # Test frame indices
    conv = FrameSamplingFilter(every=3)
    derived = conv.transform(video)
    assert derived.n_frames == 35
    assert derived.frame_index[4] == 12
    conv = FrameSamplingFilter(hertz=3)
    derived = conv.transform(video)
    assert derived.n_frames == 27
    assert derived.frame_index[3] == 12
    conv = FrameSamplingFilter(hertz=24)
    derived = conv.transform(video)
    assert derived.n_frames == 210
    assert derived.frame_index[4] == 2
    video.fps = 11.8
    conv = FrameSamplingFilter(hertz=1)
    derived = conv.transform(video)
    assert derived.n_frames == 9
    assert derived.frame_index[4] == 47
    assert derived.frame_index[5] == 59


def test_frame_sampling_cv2():
    pytest.importorskip('cv2')
    filename = join(VIDEO_DIR, 'small.mp4')
    video = VideoStim(filename)

    conv = FrameSamplingFilter(top_n=5)
    derived = conv.transform(video)
    assert derived.n_frames == 5
    assert type(next(f for f in derived)) == VideoFrameStim


def test_video_trimming_filter():
    video = VideoStim(join(VIDEO_DIR, 'small.mp4'))
    filt = TemporalTrimmingFilter(end=4.0)
    short_video = filt.transform(video)
    assert short_video.fps == 30
    assert short_video.duration == 4.0

    frame_filt = VideoTrimmingFilter(end=100, frames=True)
    short_video = frame_filt.transform(video)
    assert short_video.fps == 30
    assert short_video.n_frames == 100

    error_filt = VideoTrimmingFilter(end=10.0, validation='strict')
    with pytest.raises(ValueError):
        short_video = error_filt.transform(video)
