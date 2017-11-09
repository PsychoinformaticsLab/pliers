from os.path import join
from ..utils import get_test_data_path
from pliers.extractors import FarnebackOpticalFlowExtractor
from pliers.stimuli import VideoStim
import numpy as np
import pytest

VIDEO_DIR = join(get_test_data_path(), 'video')


def test_optical_flow_extractor():
    pytest.importorskip('cv2')
    stim = VideoStim(join(VIDEO_DIR, 'small.mp4'), onset=4.2)
    result = FarnebackOpticalFlowExtractor().transform(stim).to_df()
    target = result.query('onset==7.2')['total_flow']
    # Value returned by cv2 seems to change over versions, so use low precision
    assert np.isclose(target, 86248.05, 1e-4)
