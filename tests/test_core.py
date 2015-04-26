from unittest import TestCase
from utils import get_test_data_path
from annotations.stims import VideoStim, VideoFrameStim, ComplexTextStim
from annotations.annotators import (FaceDetectionAnnotator,
                                    DenseOpticalFlowAnnotator,
                                    VideoAnnotator, TextDictionaryAnnotator)
from annotations.io import FSLExporter, TimelineExporter
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
        timeline = stim.annotate(annotators)
        exp = FSLExporter()
        tmpdir = tempfile.mkdtemp()
        exp.export(timeline, tmpdir)
        from glob import glob
        files = glob(join(tmpdir, '*.txt'))
        self.assertEquals(len(files), 2)
        shutil.rmtree(tmpdir)

    def test_complex_text_stim(self):
        text_dir = join(get_test_data_path(), 'text')
        stim = ComplexTextStim(join(text_dir, 'complex_stim_no_header.txt'),
                               columns='ot', default_duration=0.2)
        self.assertEquals(len(stim.elements), 4)
        self.assertEquals(stim.elements[2].onset, 34)
        self.assertEquals(stim.elements[2].duration, 0.2)
        stim = ComplexTextStim(join(text_dir, 'complex_stim_with_header.txt'))
        self.assertEquals(len(stim.elements), 4)
        self.assertEquals(stim.elements[2].duration, 0.1)

    def test_text_annotation(self):
        text_dir = join(get_test_data_path(), 'text')
        stim = ComplexTextStim(join(text_dir, 'sample_text.txt'),
                               columns='to', default_duration=1)
        td = TextDictionaryAnnotator(join(text_dir,
                                          'test_lexical_dictionary.txt'),
                                     variables=['length', 'frequency'])
        self.assertEquals(td.data.shape, (7, 2))
        timeline = stim.annotate([td])
        df = TimelineExporter.timeline_to_df(timeline)
        self.assertEquals(df.shape, (12, 4))
        self.assertEquals(df.iloc[9, 3], 10.6)
