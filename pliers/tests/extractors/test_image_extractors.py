from os.path import join
from ..utils import get_test_data_path
from pliers.extractors import (BrightnessExtractor,
                               SharpnessExtractor,
                               VibranceExtractor,
                               SaliencyExtractor,
                               TensorFlowInceptionV3Extractor,
                               FaceRecognitionFaceLandmarksExtractor,
                               FaceRecognitionFaceLocationsExtractor,
                               FaceRecognitionFaceEncodingsExtractor)
from pliers.stimuli import ImageStim
from pliers.extractors.base import merge_results
import numpy as np
import pytest

IMAGE_DIR = join(get_test_data_path(), 'image')


def test_brightness_extractor():
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'), onset=4.2, duration=1)
    result = BrightnessExtractor().transform(stim).to_df()
    brightness = result['brightness'][0]
    assert np.isclose(brightness, 0.88784294, 1e-5)
    assert result['onset'][0] == 4.2
    assert result['duration'][0] == 1


def test_sharpness_extractor():
    pytest.importorskip('cv2')
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'), onset=4.2, duration=1)
    result = SharpnessExtractor().transform(stim).to_df()
    sharpness = result['sharpness'][0]
    assert np.isclose(sharpness, 1.0, 1e-5)
    assert result['onset'][0] == 4.2
    assert result['duration'][0] == 1


def test_vibrance_extractor():
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'), onset=4.2, duration=1)
    result = VibranceExtractor().transform(stim).to_df()
    color = result['vibrance'][0]
    assert np.isclose(color, 1370.65482988, 1e-5)
    assert result['onset'][0] == 4.2
    assert result['duration'][0] == 1


def test_saliency_extractor():
    pytest.importorskip('cv2')
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    result = SaliencyExtractor().transform(stim).to_df()
    ms = result['max_saliency'][0]
    assert np.isclose(ms, 0.99669953, 1e-5)
    sf = result['frac_high_saliency'][0]
    assert np.isclose(sf, 0.27461971, 1e-5)


def test_tensor_flow_inception_v3_extractor():
    imgs = [join(IMAGE_DIR, f) for f in ['apple.jpg', 'obama.jpg']]
    imgs = [ImageStim(im, onset=4.2, duration=1) for im in imgs]
    ext = TensorFlowInceptionV3Extractor()
    results = ext.transform(imgs)
    df = merge_results(results, format='wide', extractor_names='multi')
    assert len(df) == 2
    assert ('TensorFlowInceptionV3Extractor', 'Granny Smith') in df.columns
    assert 0.22610 in df[
        ('TensorFlowInceptionV3Extractor', 'Windsor tie')].values
    assert 4.2 in df[('onset', np.nan)].values
    assert 1 in df[('duration', np.nan)].values


def test_face_recognition_landmarks_extractor():
    pytest.importorskip('face_recognition')
    ext = FaceRecognitionFaceLandmarksExtractor()
    imgs = [join(IMAGE_DIR, f) for f in ['apple.jpg', 'thai_people.jpg',
                                         'obama.jpg']]
    result = ext.transform(imgs)
    dfs = [r.to_df(timing=False) for r in result]
    assert dfs[0].empty
    assert dfs[1].shape == (4, 10)
    assert dfs[2].shape == (1, 10)
    assert 'face_landmarks_nose_tip' in dfs[1].columns
    assert 'face_landmarks_nose_tip' in dfs[2].columns
    assert dfs[1].loc[3, 'face_landmarks_left_eyebrow'] == \
        result[1].raw[3]['left_eyebrow']


def test_face_recognition_encodings_extractor():
    pytest.importorskip('face_recognition')
    ext = FaceRecognitionFaceEncodingsExtractor()
    imgs = [join(IMAGE_DIR, f) for f in ['apple.jpg', 'thai_people.jpg',
                                         'obama.jpg']]
    result = ext.transform(imgs)
    dfs = [r.to_df(timing=False) for r in result]
    assert dfs[0].empty
    assert dfs[1].iloc[0, 1].shape == (128,)
    assert dfs[2].iloc[0, 1].shape == (128,)
    assert 'object_id' in dfs[1].columns
    assert dfs[1]['object_id'][1] == 1
    assert 'face_encodings' in dfs[1].columns
    assert 'face_encodings' in dfs[2].columns


def test_face_recognition_locations_extractor():
    pytest.importorskip('face_recognition')
    ext = FaceRecognitionFaceLocationsExtractor()
    imgs = [join(IMAGE_DIR, f) for f in ['apple.jpg', 'thai_people.jpg',
                                         'obama.jpg']]
    result = ext.transform(imgs)
    dfs = [r.to_df(timing=False) for r in result]
    assert dfs[0].empty
    assert isinstance(dfs[1].iloc[0, 1], tuple)
    assert len(dfs[1].iloc[0, 1]) == 4
    assert len(dfs[2].iloc[0, 1]) == 4
    assert dfs[1]['object_id'][1] == 1
    assert 'face_locations' in dfs[1].columns
    assert 'face_locations' in dfs[2].columns
