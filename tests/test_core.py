from unittest import TestCase
from utils import get_test_data_path
from annotations.stims import VideoStim, VideoFrameStim
from os.path import join
import numpy as np


class TestCore(TestCase):

    def test_video_stim(self):
        ''' Test VideoStim functionality. '''
        filename = join(get_test_data_path(), 'video', 'small.mp4')
        video = VideoStim(filename, 'test video')
        self.assertEquals(video.fps, 30)
        self.assertEquals(video.n_frames, 166)
        self.assertEquals(video.width, 560)

        # Test frame iterator
        frames = [f for f in video]
        self.assertEquals(len(frames), 166)
        f = frames[100]
        self.assertIsInstance(f, VideoFrameStim)
        self.assertIsInstance(f.onset, float)
        self.assertEquals(f.data.shape, (320, 560, 3))
