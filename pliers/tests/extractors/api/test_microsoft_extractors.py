from pliers.extractors import (MicrosoftAPIFaceExtractor,
                               MicrosoftAPIFaceEmotionExtractor,
                               MicrosoftVisionAPIExtractor,
                               MicrosoftVisionAPITagExtractor,
                               MicrosoftVisionAPICategoryExtractor,
                               MicrosoftVisionAPIImageTypeExtractor,
                               MicrosoftVisionAPIColorExtractor,
                               MicrosoftVisionAPIAdultExtractor)
from pliers.stimuli import ImageStim
import pytest
from os.path import join
from ...utils import get_test_data_path

IMAGE_DIR = join(get_test_data_path(), 'image')


@pytest.mark.requires_payment
@pytest.mark.skipif("'MICROSOFT_FACE_SUBSCRIPTION_KEY' not in os.environ")
def test_microsoft_api_face_extractor():
    ext = MicrosoftAPIFaceExtractor()
    img = ImageStim(join(IMAGE_DIR, 'thai_people.jpg'))
    res = ext.transform(img).to_df()
    assert res.shape == (4, 8)
    assert 'faceRectangle_width' in res.columns

    all_attr = ['accessories', 'age', 'blur', 'emotion', 'exposure',
                'facialHair', 'gender', 'glasses', 'hair', 'headPose',
                'makeup', 'noise', 'occlusion', 'smile']
    ext = MicrosoftAPIFaceExtractor(face_id=True, landmarks=True,
                                    attributes=all_attr)
    img = ImageStim(join(IMAGE_DIR, 'CC0', '21748993669_8b41319d6f_z.jpg'))
    res = ext.transform(img).to_df()
    assert res.shape == (2, 101)
    assert 'faceRectangle_width' in res.columns
    assert 'faceId' in res.columns
    assert 'faceLandmarks_eyeLeftBottom_x' in res.columns
    assert 'face_hair_invisible' in res.columns
    assert res['face_hair_invisible'][0] != res['face_hair_invisible'][1]
    assert 'face_gender' in res.columns
    assert res['face_gender'][0] == 'female'


@pytest.mark.requires_payment
@pytest.mark.skipif("'MICROSOFT_FACE_SUBSCRIPTION_KEY' not in os.environ")
def test_microsoft_api_face_emotion_extractor():
    ext = MicrosoftAPIFaceEmotionExtractor()
    img = ImageStim(join(IMAGE_DIR, 'obama.jpg'))
    res = ext.transform(img).to_df(timing=False, object_id=False)
    assert res.shape == (1, 8)
    assert res['face_emotion_happiness'][0] > 0.5
    assert res['face_emotion_anger'][0] < 0.5


@pytest.mark.requires_payment
@pytest.mark.skipif("'MICROSOFT_VISION_SUBSCRIPTION_KEY' not in os.environ")
def test_microsoft_vision_api_extractor():
    ext = MicrosoftVisionAPIExtractor()
    img = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    res = ext.transform(img).to_df(timing=False, object_id=False)
    assert res.shape == (1, 16)
    assert 'apple' in res.columns
    assert 'lineDrawingType' in res.columns
    assert 'dominantColors' in res.columns
    assert 'isAdultContent' in res.columns

    ext = MicrosoftVisionAPIExtractor(features=['Color', 'Tags'])
    res = ext.transform(img).to_df(timing=False, object_id=False)
    assert res.shape == (1, 10)
    assert 'apple' in res.columns
    assert 'lineDrawingType' not in res.columns
    assert 'dominantColors' in res.columns
    assert 'isAdultContent' not in res.columns


@pytest.mark.requires_payment
@pytest.mark.skipif("'MICROSOFT_VISION_SUBSCRIPTION_KEY' not in os.environ")
def test_microsoft_vision_api_tag_extractor():
    ext = MicrosoftVisionAPITagExtractor()
    img = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    res = ext.transform(img).to_df(timing=False, object_id=False)
    assert 'fruit' in res.columns
    assert 'apple' in res.columns
    assert res['apple'][0] > 0.7


@pytest.mark.requires_payment
@pytest.mark.skipif("'MICROSOFT_VISION_SUBSCRIPTION_KEY' not in os.environ")
def test_microsoft_vision_api_category_extractor():
    ext = MicrosoftVisionAPICategoryExtractor()
    img = ImageStim(join(IMAGE_DIR, 'obama.jpg'))
    res = ext.transform(img).to_df(timing=False, object_id=False)
    assert res.shape == (1, 1)
    assert 'people_portrait' in res.columns
    assert res['people_portrait'][0] > 0.5


@pytest.mark.requires_payment
@pytest.mark.skipif("'MICROSOFT_VISION_SUBSCRIPTION_KEY' not in os.environ")
def test_microsoft_vision_api_type_extractor():
    ext = MicrosoftVisionAPIImageTypeExtractor()
    img = ImageStim(join(IMAGE_DIR, 'obama.jpg'))
    res = ext.transform(img).to_df()
    assert res.shape == (1, 6)
    assert res['clipArtType'][0] == 0


@pytest.mark.requires_payment
@pytest.mark.skipif("'MICROSOFT_VISION_SUBSCRIPTION_KEY' not in os.environ")
def test_microsoft_vision_api_color_extractor():
    ext = MicrosoftVisionAPIColorExtractor()
    img = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    res = ext.transform(img).to_df(timing=False, object_id=False)
    assert res.shape == (1, 5)
    assert res['dominantColorForeground'][0] == 'Red'
    assert res['dominantColorBackground'][0] == 'White'
    assert not res['isBwImg'][0]


@pytest.mark.requires_payment
@pytest.mark.skipif("'MICROSOFT_VISION_SUBSCRIPTION_KEY' not in os.environ")
def test_microsoft_vision_api_adult_extractor():
    ext = MicrosoftVisionAPIAdultExtractor()
    img = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    res = ext.transform(img).to_df(timing=False, object_id=False)
    assert res.shape == (1, 4)
    assert res['adultScore'][0] < 0.5

    img = ImageStim(join(IMAGE_DIR, 'CC0', '3333259349_0177d46bbf_z.jpg'))
    res = ext.transform(img).to_df(timing=False, object_id=False)
    assert res['adultScore'][0] > 0.1
    assert res['racyScore'][0] > 0.1
