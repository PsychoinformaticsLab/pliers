from unittest import TestCase
from utils import get_test_data_path
from annotations.stims import VideoStim, VideoFrameStim
from annotations.annotators import (FaceDetectionAnnotator,
                                    DenseOpticalFlowAnnotator,
                                    VideoAnnotator)
from annotations.io import FSLExporter
from os.path import join
import tempfile
import shutil
import pandas as pd
import cProfile


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

    def test_full_pipeline(self):
        ''' Smoke test of entire pipeline, from stimulus loading to
        event file export. '''
        stim = VideoStim(join(get_test_data_path(), 'video', 'small.mp4'))
        annotators = [DenseOpticalFlowAnnotator(), FaceDetectionAnnotator()]
        va = VideoAnnotator(annotators)
        timeline = va.apply(stim)
        exp = FSLExporter()
        tmpdir = tempfile.mkdtemp()
        exp.export(timeline, tmpdir)
        from glob import glob
        files = glob(join(tmpdir, '*.txt'))
        self.assertEquals(len(files), 2)
        shutil.rmtree(tmpdir)
