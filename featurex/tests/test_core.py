from .utils import get_test_data_path
from featurex.stimuli.image import ImageStim
from featurex.export import FSLExporter
from featurex.lazy import extract
from featurex.transformers import get_transformer
from featurex.extractors import Extractor
from featurex.extractors.image import VibranceExtractor
from featurex.extractors.audio import STFTExtractor
from os.path import join
import tempfile
import shutil
import pandas as pd


def test_full_pipeline():
    ''' Smoke test of entire pipeline, from stimulus loading to
    event file export. '''
    stim = ImageStim(join(get_test_data_path(), 'image', 'obama.jpg'))
    timeline = VibranceExtractor().extract(stim).to_df(stim_name=True)
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
    assert isinstance(results[0], pd.DataFrame)


def test_get_transformer_by_name():
    tda = get_transformer('stFteXtrActOr', base=Extractor)
    assert isinstance(tda, STFTExtractor)
