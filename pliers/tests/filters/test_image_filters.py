from os.path import join
from ..utils import get_test_data_path
from pliers.filters import (ImageCroppingFilter,
                            ImageResizingFilter,
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


def test_image_resizing_filter():
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    size = (299, 299)
    filt = ImageResizingFilter(size=size, maintain_aspect_ratio=False)
    new_stim = filt.transform(stim)
    # Test that only 2 pixels are black.
    assert (new_stim.data == 0).all(-1).sum() < 100

    filt2 = ImageResizingFilter(size=size, maintain_aspect_ratio=True)
    new_stim = filt2.transform(stim)
    assert new_stim.data.shape == (*size, 3)
    # Test that many pixels are now black, because of the black border that is
    # introduced when maintaining aspect ratio.
    assert (new_stim.data == 0).all(-1).sum() > 25000

    assert ImageResizingFilter((24, 24), resample='nearest').resample == 0
    assert ImageResizingFilter((24, 24), resample='bilinear').resample == 2
    assert ImageResizingFilter((24, 24), resample='bicubic').resample == 3
    assert ImageResizingFilter((24, 24), resample='lanczos').resample == 1
    assert ImageResizingFilter((24, 24), resample='box').resample == 4
    assert ImageResizingFilter((24, 24), resample='hamming').resample == 5
    with pytest.raises(ValueError):
        ImageResizingFilter((24, 24), resample='farthest')


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
