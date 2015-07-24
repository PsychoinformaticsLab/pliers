from unittest import TestCase
from utils import _get_test_data_path
from annotations.stims import VideoStim
from annotations.annotators.image import FaceDetectionAnnotator
from annotations.annotators.video import DenseOpticalFlowAnnotator
from annotations.io import FSLExporter
from os.path import join
import tempfile
import shutil


class TestCore(TestCase):

    def test_full_pipeline(self):
        ''' Smoke test of entire pipeline, from stimulus loading to
        event file export. '''
        stim = VideoStim(join(_get_test_data_path(), 'video', 'small.mp4'))
        annotators = [DenseOpticalFlowAnnotator(), FaceDetectionAnnotator()]
        timeline = stim.annotate(annotators, show=False)
        exp = FSLExporter()
        tmpdir = tempfile.mkdtemp()
        exp.export(timeline, tmpdir)
        from glob import glob
        files = glob(join(tmpdir, '*.txt'))
        self.assertEquals(len(files), 2)
        shutil.rmtree(tmpdir)
