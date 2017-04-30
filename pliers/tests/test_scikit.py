from .utils import get_test_data_path

import pytest
import numpy as np

from os.path import join
from pliers.extractors import BrightnessExtractor, SharpnessExtractor
from pliers.graph import Graph
from pliers.scikit import PliersTransformer
from pliers.stimuli import ImageStim
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


def test_graph_scikit():
    pytest.importorskip('pytesseract')
    image_dir = join(get_test_data_path(), 'image')
    stim1 = join(image_dir, 'apple.jpg')
    stim2 = join(image_dir, 'button.jpg')
    graph_spec = join(get_test_data_path(), 'graph', 'simple_graph.json')
    graph = Graph(spec=graph_spec)
    trans = PliersTransformer(graph)
    res = trans.fit_transform([stim1, stim2])
    assert res.shape == (2, 1)
    assert res[0][0] == 0
    assert res[1][0] == 4


def test_extractor_scikit():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    ext = BrightnessExtractor()
    trans = PliersTransformer(ext)
    res = trans.fit_transform(stim)
    assert res.shape == (1, 1)
    assert np.isclose(res[0][0], 0.88784294, 1e-5)


def test_within_pipeline():
    pytest.importorskip('cv2')
    stim = join(get_test_data_path(), 'image', 'apple.jpg')
    graph = Graph([BrightnessExtractor(), SharpnessExtractor()])
    trans = PliersTransformer(graph)
    normalizer = Normalizer()
    pipeline = Pipeline([('pliers', trans), ('normalizer', normalizer)])
    res = pipeline.fit_transform(stim)
    assert res.shape == (1, 2)
    assert np.isclose(res[0][0], 0.66393, 1e-5)
    assert np.isclose(res[0][1], 0.74780, 1e-5)
