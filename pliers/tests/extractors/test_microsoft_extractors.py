from pliers.extractors import (MicrosoftAPIFaceExtractor,
                               MicrosoftAPIFaceEmotionExtractor,
                               MicrosoftVisionAPIExtractor,
                               MicrosoftVisionAPITagExtractor,
                               MicrosoftVisionAPICategoryExtractor,
                               MicrosoftVisionAPIImageTypeExtractor,
                               MicrosoftVisionAPIColorExtractor,
                               MicrosoftVisionAPIAdultExtractor,
                               ExtractorResult,
                               merge_results)
from pliers.stimuli import ImageStim
import pytest
from os.path import join
from ..utils import get_test_data_path

IMAGE_DIR = join(get_test_data_path(), 'image')


@pytest.mark.skipif("'MICROSOFT_FACE_SUBSCRIPTION_KEY' not in os.environ")
def test_microsoft_api_face_extractor():
    ext = MicrosoftAPIFaceExtractor()
    img = ImageStim(join(IMAGE_DIR, 'thai_people.jpg'))
    res = ext.transform(img).to_df()
    assert res.shape == (4, 8)
    assert 'faceRectangle_width' in res.columns

    # all_attr = ['accessories', 'age', 'blur', 'emotion', 'exposure',
    #             'facialHair', 'gender', 'glasses', 'hair', 'headPose',
    #             'makeup', 'noise', 'occlusion', 'smile']
    # ext = MicrosoftAPIFaceExtractor(face_id=True, landmarks=True,
    #                                 attributes=all_attr)
    # img = ImageStim(join(IMAGE_DIR, 'CC0', '21748993669_8b41319d6f_z.jpg'))
    # res = ext.transform(img).to_df()
    # assert False
