''' Filters that operate on ImageStim inputs. '''

from pliers.stimuli.image import ImageStim
from PIL import Image
from .base import Filter


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
                         data=new_img,
                         onset=stim.onset,
                         duration=stim.duration)
