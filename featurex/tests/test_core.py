from .utils import get_test_data_path
from featurex.stimuli.video import VideoStim
from featurex.extractors.image import VibranceExtractor
from featurex.export import FSLExporter
from featurex.core import Timeline, Value, Event
from featurex.lazy import extract
from featurex.transformers import get_transformer
from featurex.extractors import Extractor
from featurex.extractors.audio import STFTExtractor
from os.path import join
import tempfile
import shutil


def test_full_pipeline():
    ''' Smoke test of entire pipeline, from stimulus loading to
    event file export. '''
    stim = VideoStim(join(get_test_data_path(), 'video', 'small.mp4'))
    extractors = [VibranceExtractor()]
    timeline = stim.extract(extractors, show=False)
    exp = FSLExporter()
    tmpdir = tempfile.mkdtemp()
    exp.export(timeline, tmpdir)
    from glob import glob
    files = glob(join(tmpdir, '*.txt'))
    assert len(files) == 1
    shutil.rmtree(tmpdir)

def test_lazy_extraction():
    textfile = join(get_test_data_path(), 'text', 'scandal.txt')
    results = extract([textfile], ['lengthextractor', 'numuniquewordsextractor'])
    assert isinstance(results[0], Value)

def test_dummy_code_timeline():
    data = [{'A': 12.0, 'B': 'abc'}, { 'A': 7, 'B': 'def'}, { 'C': 40 }]
    events = [Event(values=[Value(None, None, x)], duration=1) for x in data]
    tl = Timeline(events=events, period=1)
    assert tl.to_df().shape == (5, 4)
    tl_dummy = tl.dummy_code()
    assert tl_dummy.to_df().shape == (7, 4)
    tl = Timeline(events=events, period=1)
    tl_dummy = tl.dummy_code(string_only=False)
    assert tl_dummy.to_df().shape == (9, 4)

def test_get_transformer_by_name():
    tda = get_transformer('stFteXtrActOr', base=Extractor)
    assert isinstance(tda, STFTExtractor)
