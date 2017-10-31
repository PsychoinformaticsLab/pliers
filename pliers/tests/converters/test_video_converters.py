from os.path import join
from ..utils import get_test_data_path
from pliers.converters import VideoToAudioConverter
from pliers.stimuli import VideoStim
import numpy as np

VIDEO_DIR = join(get_test_data_path(), 'video')


def test_video_to_audio_converter():
    filename = join(VIDEO_DIR, 'small.mp4')
    video = VideoStim(filename, onset=4.2)
    conv = VideoToAudioConverter()
    audio = conv.transform(video)
    assert audio.history.source_class == 'VideoStim'
    assert audio.history.source_file == filename
    assert audio.onset == 4.2
    assert np.isclose(video.duration, audio.duration, 1e-2)
