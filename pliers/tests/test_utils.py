from types import GeneratorType
from os.path import join
import numpy as np

import pytest

from pliers.stimuli import VideoStim
from pliers.filters import FrameSamplingFilter
from pliers.utils import batch_iterable, flatten_dict, resample
from pliers.extractors import RMSExtractor
from pliers import config

from .utils import get_test_data_path


@pytest.mark.skip(reason="tqdm prevents normal stdout/stderr capture; need to"
                  "figure out why.")
def test_progress_bar(capfd):

    video_dir = join(get_test_data_path(), 'video')
    video = VideoStim(join(video_dir, 'obama_speech.mp4'))
    conv = FrameSamplingFilter(hertz=2)

    old_val = config.get_option('progress_bar')
    config.set_option('progress_bar', True)

    derived = conv.transform(video)
    out, err = capfd.readouterr()
    assert 'Video frame:' in err and '100%' in err

    config.set_option('progress_bar', False)
    derived = conv.transform(video)
    out, err = capfd.readouterr()
    assert 'Video frame:' not in err and '100%' not in err

    config.set_option('progress_bar', old_val)


def test_batch_iterable():
    iterable = [1, 2, 3, 4, 5, 6, 7, 8]
    res = batch_iterable(iterable, 2)
    assert isinstance(res, GeneratorType)
    assert sum(1 for i in res) == 4
    res = batch_iterable(iterable, 4)
    assert sum(1 for i in res) == 2
    res = batch_iterable(iterable, 4)
    first_half = next(res)
    assert isinstance(first_half, list)
    assert first_half == [1, 2, 3, 4]
    second_half = next(res)
    assert isinstance(second_half, list)
    assert second_half == [5, 6, 7, 8]


def test_flatten_dict():
    d = { 'a' : 5, 'b' : { 'c' : 6, 'd' : 1 } }
    res = flatten_dict(d)
    assert res == { 'a' : 5, 'b_c' : 6, 'b_d' : 1}
    res = flatten_dict(d, 'prefix', '.')
    assert res == { 'prefix.a' : 5, 'prefix.b.c' : 6, 'prefix.b.d' : 1}


def test_resample():
    ext = RMSExtractor()
    res = ext.transform(join(get_test_data_path(), 'audio/homer.wav'))

    df = res.to_df(format='long')

    # Test downsample
    downsampled_df = resample(df, 3)

    assert np.allclose(downsampled_df.ix[0].onset, 0)
    assert np.allclose(downsampled_df.ix[1].onset, 0.33333)

    assert set(downsampled_df.columns) == {
        'duration', 'onset', 'feature', 'value'}

    assert downsampled_df['feature'].unique() == 'rms'

    # This checks that the filtering has happened. If it has not, then
    # this value for this frequency bin will be an alias and have a
    # very different amplitude
    assert downsampled_df[downsampled_df.onset == 0]['value'].values[0] != \
        df[df.onset == 0]['value'].values[0]
    assert downsampled_df[downsampled_df.onset == 0]['value'].values[0] != \
        df[df.onset == 0]['value'].values[0]

    assert np.allclose(downsampled_df[downsampled_df.onset == 2]['value'].values[0],
                       0.2261582761938699)


    # Test upsample
    ext = RMSExtractor(frame_length=1500, hop_length=1500,)
    res = ext.transform(join(get_test_data_path(), 'audio/homer.wav'))
    df = res.to_df(format='long')

    upsampled_df = resample(df, 10)

    assert np.allclose(upsampled_df.ix[0].onset, 0)
    assert np.allclose(upsampled_df.ix[1].onset, 0.1)

    assert set(upsampled_df.columns) == {
        'duration', 'onset', 'feature', 'value'}

    assert upsampled_df['feature'].unique() == 'rms'

    # This checks that the filtering has happened. If it has not, then
    # this value for this frequency bin will be an alias and have a
    # very different amplitude
    assert upsampled_df[upsampled_df.onset == 0]['value'].values[0] != \
        df[df.onset == 0]['value'].values[0]
    assert upsampled_df[upsampled_df.onset == 0]['value'].values[0] != \
        df[df.onset == 0]['value'].values[0]

    # Value will be slightly different at 2s with different sampling
    assert np.allclose(
        upsampled_df[upsampled_df.onset == 2]['value'].values[0],  0.25309)
