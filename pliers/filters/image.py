''' Filters that operate on ImageStim inputs. '''

from pliers.stimuli.image import ImageStim
from PIL import Image
from PIL import ImageFilter as PillowFilter
from .base import Filter

import numpy as np


class ImageFilter(Filter):

    ''' Base class for all ImageFilters. '''

    _input_type = ImageStim


class ImageCroppingFilter(ImageFilter):

    ''' Crops an image.

    Args:
        box (tuple): a 4-length tuple containing the left, upper, right, and
            lower coordinates for the desired region of the image. If none is
            specified, crops out black borders from the image.
    '''

    _log_attributes = ('box',)
    VERSION = '1.0'

    def __init__(self, box=None):
        self.box = box
        super(ImageCroppingFilter, self).__init__()

    def _filter(self, stim):
        if self.box:
            x0, y0, x1, y1 = self.box
        else:
            pillow_img = Image.fromarray(stim.data)
            x0, y0, x1, y1 = pillow_img.getbbox()
        new_img = stim.data[y0:y1, x0:x1]
        return ImageStim(stim.filename,
                         data=new_img)


class ImageResizingFilter(ImageFilter):

    ''' Resizes an image, while optionally maintaining aspect ratio.

    Args:
        size (tuple of two ints): new size of the image.
        maintain_aspect_ratio (boolean): if true, resize the image while
            maintaining aspect ratio, and pad the rest with zero values.
            Otherwise, potentially distort the image during resizing to fit the
            new size.
        resample str: resampling method. One of 'nearest', 'bilinear',
            'bicubic', 'lanczos', 'box', and 'hamming'. See
            https://pillow.readthedocs.io/en/5.1.x/handbook/concepts.html#concept-filters
            for more information.
    '''

    _log_attributes = ('size', 'maintain_aspect_ratio', 'resample')
    VERSION = '1.0'

    def __init__(self, size, maintain_aspect_ratio=False, resample='bicubic'):
        self.size = size
        self.maintain_aspect_ratio = maintain_aspect_ratio
        resampling_mapping = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS,
            'box': Image.BOX,
            'hamming': Image.HAMMING,
        }
        if resample.lower() not in resampling_mapping.keys():
            raise ValueError(
                "Unknown resampling method '{}'. Allowed values are '{}'"
                .format(resample, "', '".join(resampling_mapping.keys())))
        self.resample = resampling_mapping[resample]
        super(ImageResizingFilter, self).__init__()

    def _filter(self, stim):
        pillow_img = Image.fromarray(stim.data)

        if not self.maintain_aspect_ratio:
            new_img = np.array(
                pillow_img.resize(self.size, resample=self.resample))
        else:
            # Resize the image to the requested size in one of the dimensions.
            # We then create a black image of the requested size and paste the
            # resized image into the middle of this new image. The effect is
            # that there is a black border on the top and bottom or the left
            # and right of the resized image.
            orig_size = pillow_img.size
            ratio = max(self.size) / max(orig_size)
            inter_size = (np.array(orig_size) * ratio).astype(np.int32)
            inter_img = pillow_img.resize(inter_size, resample=self.resample)
            new_img = Image.new('RGB', self.size)
            upper_left = (
                (self.size[0] - inter_size[0]) // 2,
                (self.size[1] - inter_size[1]) // 2)
            new_img.paste(inter_img, box=upper_left)
            new_img = np.array(new_img)

        return ImageStim(stim.filename, data=new_img)


class PillowImageFilter(ImageFilter):

    ''' Uses the ImageFilter module from PIL to run a pre-defined image enhancement
    filter on an ImageStim.

    Sample of available filters:
    BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE, EMBOSS, FIND_EDGES,
    SMOOTH, SMOOTH_MORE, SHARPEN

    Args:
        image_filter (str or type or ImageFilter): specific name or type of the
            filter to be used, with supporting *args and **kwargs. Also
            accepted to directly pass an instance of PIL's ImageFilter.Filter
        args, kwargs: Optional positional and keyword arguments passed onto
            the pillow ImageFilter initializer.
    '''

    _log_attributes = ('filter',)

    def __init__(self, image_filter=None, *args, **kwargs):
        if image_filter is None:
            pillow_url = "http://pillow.readthedocs.io/en/3.4.x/reference/"
            "ImageFilter.html#filters"
            raise ValueError("Must enter a valid filter to use. See %s"
                             "for a list of valid PIL filters." % pillow_url)
        if isinstance(image_filter, type):
            image_filter = image_filter(*args, **kwargs)

        if isinstance(image_filter, PillowFilter.Filter):
            self.filter = image_filter
        elif isinstance(image_filter, str):
            self.filter = getattr(PillowFilter, image_filter)(*args, **kwargs)
        else:
            raise ValueError("Must provide an image_filter as a string, type, "
                             "or ImageFilter object. ")

        super(PillowImageFilter, self).__init__()

    def _filter(self, stim):
        pillow_img = Image.fromarray(stim.data)
        new_img = np.array(pillow_img.filter(self.filter))
        return ImageStim(stim.filename,
                         data=new_img)
