''' Base implementation for all API transformers. '''

from pliers import config
from pliers.transformers import Transformer


class APITransformer(Transformer):

    def _iterate(self, stims, validation='strict', *args, **kwargs):
        # Check if we are trying to transform a large number of stimuli
        if not config.get_option('allow_large_jobs'):
            stims = list(stims)
            if len(stims) > 100:
                raise ValueError("Attempted to run an API transformation "
                                 "on %d stims, aborting. To allow "
                                 "transformation, change config option "
                                 "'allow_large_job' to True." % len(stims))

        return super(APITransformer, self)._iterate(stims, *args, **kwargs)

    def _transform(self, stim, *args, **kwargs):
        # Check if we are trying to transform a large amount of data
        if not config.get_option('allow_large_jobs'):
            if stim.duration > 60:
                raise ValueError("Attempted to run an API transformation "
                                 "on a stimulus of duration %f, aborting."
                                 "To allow this transformation, change "
                                 "config option 'allow_large_job' to "
                                 "True." % stim.duration)

        return super(APITransformer, self)._transform(stim, *args, **kwargs)
