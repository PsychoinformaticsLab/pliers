from os.path import join
from ..utils import get_test_data_path
from pliers.filters import (ImageCroppingFilter,
                            PillowImageFilter)
from pliers.stimuli import ImageStim
import numpy as np
import pytest

IMAGE_DIR = join(get_test_data_path(), 'image')


def test_image_cropping_filter():
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    filt = ImageCroppingFilter((210, 120, 260, 170))
    new_stim = filt.transform(stim)
    assert new_stim.data.shape == (50, 50, 3)
    assert np.array_equal(stim.data[0, 0], [255.0, 255.0, 255.0])
    # Top left corner goes white -> red
    assert np.array_equal(new_stim.data[0, 0], [136.0, 0.0, 0.0])

    filt2 = ImageCroppingFilter()
    new_stim = filt2.transform(stim)
    assert new_stim.data.shape == (288, 420, 3)
    stim2 = ImageStim(join(IMAGE_DIR, 'aspect_ratio_fail.jpg'))
    assert stim2.data.shape == (240, 240, 3)
    new_stim2 = filt2.transform(stim2)
    assert new_stim2.data.shape == (112, 240, 3)


def test_pillow_image_filter_filter():
    stim = ImageStim(join(IMAGE_DIR, 'thai_people.jpg'))
    with pytest.raises(ValueError):
        filt = PillowImageFilter()
    filt = PillowImageFilter('BLUR')
    blurred = filt.transform(stim)
    assert blurred is not None

    from PIL import ImageFilter
    filt2 = PillowImageFilter(ImageFilter.FIND_EDGES)
    edges = filt2.transform(stim)
    assert np.array_equal(edges.data[0, 0], [134, 85, 45])

    filt3 = PillowImageFilter(ImageFilter.MinFilter(3))
    min_img = filt3.transform(stim)
    assert np.array_equal(min_img.data[0, 0], [122, 74, 36])

    filt4 = PillowImageFilter('MinFilter')
    min_img = filt4.transform(stim)
    assert np.array_equal(min_img.data[0, 0], [122, 74, 36])

    filt5 = PillowImageFilter(ImageFilter.MaxFilter, size=3)
    med_img = filt5.transform(stim)
    assert np.array_equal(med_img.data[0, 0], [136, 86, 49])
