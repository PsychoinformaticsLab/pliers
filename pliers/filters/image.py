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
