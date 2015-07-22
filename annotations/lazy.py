from .core import Timeline, Note, Event
from .stims import VideoStim, ImageStim
from .io import load

def annotate(stims, annotations, format='df'):
    """ Annotate one or more stims using one or annotations.
    Args:
        stims (list): A list of strings pointing to the filenames of stims.
        annotations (list): A list of strings giving the names of the desired
            annotations to apply to each stim.
        format (str): The format to return the data in.
    """
    stims = load(stims)
    for s in stims:
        res = s.annotate(annotations)