from .stimuli import Stim, load_stims
from .transformers import Transformer, get_transformer


def extract(stims, transformers, format='df'):
    """ Feature extraction, the lazy way. Pass a list of stimuli and a list of
    transformers, then go have a drink.
    Args:
        stims (list): A list of stims to process. Each element can be either
            a string (a single filename, or a directory containing stims),
            or a Stim object.
        transformers (list): A list of transformers to apply to each Stim. Each
            element can be either a case-insensitive string giving the name of
            the desired transformer (which will be matched against all
                currently available transformers), or an Transformer object.
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

    transformers = [a if isinstance(a, Transformer) else get_transformer(a)
                  for a in transformers]

    return [s.extract(transformers) for s in _stims]
