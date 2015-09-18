from .stims import Stim
from .extractors import Extractor, get_extractor
from .io import load_stims


def extract(stims, extractors, format='df'):
    """ Feature extraction, the lazy way. Pass a list of stimuli and a list of
    extractors, then go have a drink.
    Args:
        stims (list): A list of stims to process. Each element can be either
            a string (a single filename, or a directory containing stims),
            or a Stim object.
        extractors (list): A list of extractors to apply to each Stim. Each
            element can be either a case-insensitive string giving the name of
            the desired extractor (which will be matched against all currently
                available extractors), or an Extractor object.
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

    extractors = [a if isinstance(a, Extractor) else get_extractor(a)
                  for a in extractors]

    return [s.extract(extractors) for s in _stims]
