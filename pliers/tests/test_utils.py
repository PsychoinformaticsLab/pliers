from pliers.stimuli import VideoStim
from pliers.filters import FrameSamplingFilter
from pliers.utils import batch_iterable, flatten_dict
from pliers import config
from types import GeneratorType
from .utils import get_test_data_path
from os.path import join
import pytest


@pytest.mark.skip(reason="tqdm prevents normal stdout/stderr capture; need to"
                  "figure out why.")
def test_progress_bar(capfd):

    video_dir = join(get_test_data_path(), 'video')
    video = VideoStim(join(video_dir, 'obama_speech.mp4'))
    conv = FrameSamplingFilter(hertz=2)

    old_val = config.get_option('progress_bar')
    config.set_option('progress_bar', True)

    derived = conv.transform(video)
    out, err = capfd.readouterr()
    assert 'Video frame:' in err and '100%' in err

    config.set_option('progress_bar', False)
    derived = conv.transform(video)
    out, err = capfd.readouterr()
    assert 'Video frame:' not in err and '100%' not in err

    config.set_option('progress_bar', old_val)


def test_batch_iterable():
    iterable = [1, 2, 3, 4, 5, 6, 7, 8]
    res = batch_iterable(iterable, 2)
    assert isinstance(res, GeneratorType)
    assert sum(1 for i in res) == 4
    res = batch_iterable(iterable, 4)
    assert sum(1 for i in res) == 2
    res = batch_iterable(iterable, 4)
    first_half = next(res)
    assert isinstance(first_half, list)
    assert first_half == [1, 2, 3, 4]
    second_half = next(res)
    assert isinstance(second_half, list)
    assert second_half == [5, 6, 7, 8]


def test_flatten_dict():
    d = { 'a' : 5, 'b' : { 'c' : 6, 'd' : 1 } }
    res = flatten_dict(d)
    assert res == { 'a' : 5, 'b_c' : 6, 'b_d' : 1}
    res = flatten_dict(d, 'prefix', '.')
    assert res == { 'prefix.a' : 5, 'prefix.b.c' : 6, 'prefix.b.d' : 1}
