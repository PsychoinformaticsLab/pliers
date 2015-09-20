from unittest import TestCase
from .utils import _get_test_data_path
from featurex.stimuli.video import VideoStim
from featurex.extractors.image import FaceDetectionExtractor
from featurex.export import FSLExporter
from featurex.core import Timeline
from featurex.lazy import extract
from os.path import join
import tempfile
import shutil


class TestCore(TestCase):

    def test_full_pipeline(self):
        ''' Smoke test of entire pipeline, from stimulus loading to
        event file export. '''
        stim = VideoStim(join(_get_test_data_path(), 'video', 'small.mp4'))
        extractors = [FaceDetectionExtractor()]
        timeline = stim.extract(extractors, show=False)
        exp = FSLExporter()
        tmpdir = tempfile.mkdtemp()
        exp.export(timeline, tmpdir)
        from glob import glob
        files = glob(join(tmpdir, '*.txt'))
        self.assertEquals(len(files), 1)
        shutil.rmtree(tmpdir)

    def test_lazy_extraction(self):
        stims = [join(_get_test_data_path(), 'video', 'small.mp4')]
        extractors = ['facedetectionextractor']
        results = extract(stims, extractors)
        assert isinstance(results[0], Timeline)
        textfile = join(_get_test_data_path(), 'text', 'scandal.txt')
        results = extract([textfile], ['basicstatsextractorcollection'])
