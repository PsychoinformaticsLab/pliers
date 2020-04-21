from pliers.extractors import (VectorMetricExtractor)
from pliers.stimuli import VectorStim
from pliers.tests.utils import get_test_data_path
import numpy as np
import scipy
from os.path import join
import pytest

VECTOR_DIR = join(get_test_data_path(), 'vector')

def test_vector_metric_extractor():

    def dummy(array):
        return array[0]

    def dummy_list(array):
        return array[0], array[1]

    str_func = 'lambda x: np.mean(x) ** 2'

    json_filename=join(VECTOR_DIR, 'vector_dict.json')
    stim = VectorStim(array=np.linspace(1., 4., 20), onset=2., duration=.5)
    stim_file = VectorStim(filename=json_filename)

    ext_single = VectorMetricExtractor(functions='np.mean')
    ext_multiple = VectorMetricExtractor(functions=['np.mean', 'numpy.min', 
                                         scipy.stats.entropy, dummy,
                                         dummy_list, str_func])
    ext_names = VectorMetricExtractor(functions=['np.mean', 'numpy.min', 
                                         scipy.stats.entropy, dummy,
                                         dummy_list, str_func],
                                      var_names = ['mean', 'min', 'entropy',
                                      'custom1', 'custom2', 'custom3'])

    r = ext_single.transform(stim)
    r_file = ext_single.transform(stim_file)
    r_multiple = ext_multiple.transform(stim)
    r_names = ext_names.transform(stim)

    r_df = r.to_df()
    r_file_df = r_file.to_df()
    r_multiple_df = r_multiple.to_df()
    r_long = r_multiple.to_df(format='long')
    r_names_df = r_names.to_df()

    for res in [r_df, r_file_df, r_multiple_df]:
        assert res.shape[0] == 1
    assert r_long.shape[1] == len(ext_multiple.functions)
    assert r_df['onset'][0] == 2.
    assert r_df['duration'][0] == .5
    assert r_df['mean'][0] == 2.5
    assert np.isclose(r_file_df['mean'][0], -0.655, rtol=0.001)
    assert all([m in r_multiple_df.columns for m in ['mean', 'entropy']])
    assert r_multiple_df['amin'][0] == 1.
    assert r_multiple_df['dummy'][0] == 1.
    assert r_multiple_df['dummy_list'][0][0] == np.linspace(1., 4., 20)[0]
    assert r_multiple_df['dummy_list'][0][1] == np.linspace(1., 4., 20)[1]
    assert type(r_multiple_df['dummy_list'][0]) == np.ndarray
    assert r_names_df.columns[-3] == 'custom1'
    assert r_names_df.columns[-2] == 'custom2'
    assert r_names_df.columns[-1] == 'custom3'
    assert r_names_df['custom3'][0] == 2.5 ** 2