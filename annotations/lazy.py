from .stims import Stim
from .annotators import Annotator, get_annotator
from .io import load_stims


def annotate(stims, annotators, format='df'):
    """ Stimulus annotation, the lazy way. Pass a list of stimuli and a list of
    annotations, then go have a drink.
    Args:
        stims (list): A list of stims to annotate. Each element can be either
            a string (a single filename, or a directory containing stims),
            or a Stim object.
        annotators (list): A list of annotations to apply to each Stim. Each
            element can be either a case-insensitive string giving the name of
            the desired annotation (which will be matched against all currently
                available annotators), or an Annotator object.
        format (str): The format to return the data in.
    Returns:
        A list of Timelines--one per stimulus in the original list (in the same
            order).
    """
    _stims = []
    for s in stims:
        if isinstance(s, Stim):
            _stims.append(s)
        else:
            _stims.extend(load_stims(s))

    annotators = [a if isinstance(a, Annotator) else get_annotator(a)
                  for a in annotators]

    return [s.annotate(annotators) for s in _stims]
