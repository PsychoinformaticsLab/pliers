from pliers.stimuli import VideoStim
from pliers.converters import FrameSamplingConverter
from pliers import config
from .utils import get_test_data_path
from os.path import join
import pytest


@pytest.mark.skip(reason="tqdm prevents normal stdout/stderr capture; need to"
                  "figure out why.")
def test_progress_bar(capfd):

    video_dir = join(get_test_data_path(), 'video')
    video = VideoStim(join(video_dir, 'obama_speech.mp4'))
    conv = FrameSamplingConverter(hertz=2)

    old_val = config.progress_bar
    config.progress_bar = True

    derived = conv.transform(video)
    out, err = capfd.readouterr()
    assert 'Video frame:' in err and '100%' in err

    config.progress_bar = False
    derived = conv.transform(video)
    out, err = capfd.readouterr()
    assert 'Video frame:' not in err and '100%' not in err

    config.progress_bar = old_val
