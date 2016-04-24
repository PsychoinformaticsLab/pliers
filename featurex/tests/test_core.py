from unittest import TestCase
from .utils import _get_test_data_path
from featurex.stimuli.video import VideoStim
from featurex.extractors.image import FaceDetectionExtractor
from featurex.export import FSLExporter
from featurex.core import Timeline, Value, Event
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

    def test_dummy_code_timeline(self):
        data = [{'A': 12.0, 'B': 'abc'}, { 'A': 7, 'B': 'def'}, { 'C': 40 }]
        events = [Event(values=[Value(None, None, x)], duration=1) for x in data]
        tl = Timeline(events=events, period=1)
        self.assertEqual(tl.to_df().shape, (5, 4))
        tl_dummy = tl.dummy_code()
        self.assertEqual(tl_dummy.to_df().shape, (7, 4))
        tl = Timeline(events=events, period=1)
        tl_dummy = tl.dummy_code(string_only=False)
        self.assertEqual(tl_dummy.to_df().shape, (9, 4))
