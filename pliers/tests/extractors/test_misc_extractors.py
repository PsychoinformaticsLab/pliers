from pliers.extractors import (MetricExtractor, BertLMExtractor, 
                               merge_results)
from pliers.stimuli import SeriesStim, ComplexTextStim
from pliers.tests.utils import get_test_data_path
import numpy as np
import scipy
from pathlib import Path
import pytest

def test_metric_extractor():

    def dummy(array):
        return array[0]

    def dummy_list(array):
        return array[0], array[1]

    f = Path(get_test_data_path(), 'text', 'test_lexical_dictionary.txt')
    stim = SeriesStim(data=np.linspace(1., 4., 20), onset=2., duration=.5)
    stim_file = SeriesStim(filename=f, column='frequency', sep='\t',
                           index_col='text')

    ext_single = MetricExtractor(functions='numpy.mean')
    ext_idx = MetricExtractor(functions='numpy.mean', 
                              subset_idx=['for', 'testing', 'text'])
    ext_multiple = MetricExtractor(functions=['numpy.mean', 'numpy.min', 
                                              scipy.stats.entropy, dummy,
                                              dummy_list])
    ext_names = MetricExtractor(functions=['numpy.mean', 'numpy.min', 
                                           scipy.stats.entropy, dummy,
                                           dummy_list, 'tensorflow.reduce_mean'],
                                var_names=['mean', 'min', 'entropy',
                                           'custom1', 'custom2', 'tf_mean'])

    r = ext_single.transform(stim)
    r_file = ext_single.transform(stim_file)
    r_file_idx = ext_idx.transform(stim_file)
    r_multiple = ext_multiple.transform(stim)
    r_names = ext_names.transform(stim)

    r_df = r.to_df()
    r_file_df = r_file.to_df()
    r_file_idx_df = r_file_idx.to_df()
    r_multiple_df = r_multiple.to_df()
    r_long = r_multiple.to_df(format='long')
    r_names_df = r_names.to_df()

    for res in [r_df, r_file_df, r_multiple_df]:
        assert res.shape[0] == 1
    assert r_long.shape[0] == len(ext_multiple.functions)
    assert r_df['onset'][0] == 2
    assert r_df['duration'][0] == .5
    assert r_df['mean'][0] == 2.5
    assert np.isclose(r_file_df['mean'][0], 11.388, rtol=0.001)
    assert np.isclose(r_file_idx_df['mean'][0], 12.582, rtol=0.001)
    assert all([m in r_multiple_df.columns for m in ['mean', 'entropy']])
    assert r_multiple_df['amin'][0] == 1.
    assert r_multiple_df['dummy'][0] == 1.
    assert r_multiple_df['dummy_list'][0][0] == np.linspace(1., 4., 20)[0]
    assert r_multiple_df['dummy_list'][0][1] == np.linspace(1., 4., 20)[1]
    assert type(r_multiple_df['dummy_list'][0]) == np.ndarray
    assert r_names_df.columns[-3] == 'custom1'
    assert r_names_df.columns[-2] == 'custom2'
    assert r_names_df.columns[-1] == 'tf_mean'
    assert np.isclose(r_names_df['mean'][0], r_names_df['tf_mean'][0])

def test_metric_er_as_stim():
    stim = ComplexTextStim(text = 'This is [MASK] test')
    ext_bert = BertLMExtractor(return_softmax=True)
    ext_metric = MetricExtractor(functions='numpy.sum')
    r = ext_metric.transform(ext_bert.transform(stim))
    df = merge_results(r, extractor_names=False)
    assert np.isclose(df['sum'][0], 1)
