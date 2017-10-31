from os.path import join
from ..utils import get_test_data_path, DummyExtractor
from pliers.extractors import (LengthExtractor,
                               BrightnessExtractor,
                               SharpnessExtractor,
                               VibranceExtractor)
from pliers.stimuli import (ComplexTextStim, ImageStim, VideoStim,
                            AudioStim)
from pliers.support.download import download_nltk_data
from pliers.extractors.base import ExtractorResult, merge_results
import numpy as np
import pytest

TEXT_DIR = join(get_test_data_path(), 'text')


@pytest.fixture(scope='module')
def get_nltk():
    download_nltk_data()


def test_check_target_type():
    stim = ComplexTextStim(join(TEXT_DIR, 'sample_text.txt'),
                           columns='to', default_duration=1)
    td = SharpnessExtractor()
    with pytest.raises(TypeError):
        td.transform(stim)


def test_implicit_stim_iteration():
    np.random.seed(100)
    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    de = DummyExtractor()
    results = de.transform([stim1, stim2])
    assert len(results) == 2
    assert isinstance(results[0], ExtractorResult)


def test_implicit_stim_conversion():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'button.jpg'), onset=4.2)
    ext = LengthExtractor()
    result = ext.transform(stim).to_df()
    assert 'text_length' in result.columns
    assert result['text_length'][0] == 4
    assert result['onset'][0] == 4.2


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_implicit_stim_conversion2():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'homer.wav'), onset=4.2)
    ext = LengthExtractor()
    result = ext.transform(stim)
    first_word = result[0].to_df()
    assert 'text_length' in first_word.columns
    assert first_word['text_length'][0] > 0
    assert first_word['onset'][0] >= 4.2


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_implicit_stim_conversion3():
    video_dir = join(get_test_data_path(), 'video')
    stim = VideoStim(join(video_dir, 'obama_speech.mp4'), onset=4.2)
    ext = LengthExtractor()
    result = ext.transform(stim)
    first_word = result[0].to_df()
    # The word should be "today"
    assert 'text_length' in first_word.columns
    assert first_word['text_length'][0] == 5
    assert first_word['onset'][0] >= 4.2


def test_merge_extractor_results_by_features():
    np.random.seed(100)
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))

    # Merge results for static Stims (no onsets)
    extractors = [BrightnessExtractor(), VibranceExtractor()]
    results = [e.transform(stim) for e in extractors]
    df = ExtractorResult.merge_features(results)

    de_names = ['Extractor1', 'Extractor2', 'Extractor3']
    des = [DummyExtractor(name=name) for name in de_names]
    results = [de.transform(stim) for de in des]
    df = ExtractorResult.merge_features(results)
    assert df.shape == (177, 16)
    assert df.columns.levels[1].unique().tolist() == [0, 1, 2, '']
    cols = ['onset', 'duration', 'class', 'filename', 'history', 'stim_name',
            'source_file']
    assert df.columns.levels[0].unique().tolist() == de_names + cols


def test_merge_extractor_results_by_stims():
    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    de = DummyExtractor()
    results = [de.transform(stim1), de.transform(stim2)]
    df = ExtractorResult.merge_stims(results)
    assert df.shape == (200, 10)
    assert set(df.columns.tolist()) == set(
        ['onset', 'duration', 0, 1, 2, 'stim_name', 'class', 'filename',
         'history', 'source_file'])
    assert set(df['stim_name'].unique()) == set(['obama.jpg', 'apple.jpg'])


def test_merge_extractor_results():
    np.random.seed(100)
    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    de_names = ['Extractor1', 'Extractor2', 'Extractor3']
    des = [DummyExtractor(name=name) for name in de_names]
    results = [de.transform(stim1) for de in des]
    results += [de.transform(stim2) for de in des]
    df = merge_results(results)
    assert df.shape == (354, 16)
    cols = ['onset', 'duration', 'class', 'filename', 'history', 'stim_name',
            'source_file']
    assert df.columns.levels[0].unique().tolist() == de_names + cols
    assert df.columns.levels[1].unique().tolist() == [0, 1, 2, '']
    assert set(df['stim_name'].unique()) == set(['obama.jpg', 'apple.jpg'])


def test_merge_extractor_results_flattened():
    np.random.seed(100)
    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    de_names = ['Extractor1', 'Extractor2', 'Extractor3']
    des = [DummyExtractor(name=name) for name in de_names]
    results = [de.transform(stim1) for de in des]
    results += [de.transform(stim2) for de in des]
    df = merge_results(results, flatten_columns=True)
    de_cols = ['Extractor1_0', 'Extractor1_1', 'Extractor1_2',
               'Extractor2_0', 'Extractor2_1', 'Extractor2_2',
               'Extractor3_0', 'Extractor3_1', 'Extractor3_2']
    assert df.shape == (354, 16)
    cols = ['onset', 'class', 'filename', 'history', 'stim_name', 'duration',
            'source_file']
    assert set(df.columns.unique().tolist()) == set(cols + de_cols)
