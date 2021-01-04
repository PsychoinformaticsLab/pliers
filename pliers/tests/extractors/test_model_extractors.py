from os.path import join

import numpy as np
import pytest

from ..utils import get_test_data_path
from pliers.extractors import (TensorFlowKerasApplicationExtractor)
from pliers.stimuli import ImageStim
from pliers.extractors.base import merge_results


IMAGE_DIR = join(get_test_data_path(), 'image')

def test_tensorflow_keras_application_extractor():
    imgs = [join(IMAGE_DIR, f) for f in ['apple.jpg', 'obama.jpg']]
    imgs = [ImageStim(im, onset=4.2, duration=1) for im in imgs]
    ext = TensorFlowKerasApplicationExtractor()
    results = ext.transform(imgs)
    df = merge_results(results, format='wide', extractor_names='multi')
    assert df.shape == (2, 19)
    true = 0.9737075
    pred = df['TensorFlowKerasApplicationExtractor'].loc[0, 'Granny_Smith']
    assert np.isclose(true, pred, 1e-05)
    true = 0.64234024
    pred = df['TensorFlowKerasApplicationExtractor'].loc[1, 'Windsor_tie']
    assert np.isclose(true, pred, 1e-05)
    assert 4.2 in df[('onset', np.nan)].values
    assert 1 in df[('duration', np.nan)].values
    with pytest.raises(ValueError):
        TensorFlowKerasApplicationExtractor(architecture='foo')