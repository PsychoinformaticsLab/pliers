''' Base implementation for all API transformers. '''

from pliers import config
from pliers.transformers import Transformer
from pliers.utils import isiterable, EnvironmentKeyMixin


class APITransformer(Transformer, EnvironmentKeyMixin):

    def _iterate(self, stims, validation='strict', *args, **kwargs):
        # Check if we are trying to transform a large number of stimuli
        if not config.get_option('allow_large_jobs'):
            stims = list(stims)  # TODO: better way to do this?
            if len(stims) > 100:
                raise ValueError("Attempted to run an API transformation "
                                 "on %d stims, aborting. To allow "
                                 "transformation, change config option "
                                 "'allow_large_jobs' to True." % len(stims))

        return super(APITransformer, self)._iterate(stims, *args, **kwargs)

    def _transform(self, stim, *args, **kwargs):
        # Check if we are trying to transform a large amount of data
        if not config.get_option('allow_large_jobs'):
            if not isiterable(stim) and stim.duration and stim.duration > 60:
                raise ValueError("Attempted to run an API transformation "
                                 "on a stimulus of duration %f, aborting. "
                                 "To allow this transformation, change "
                                 "config option 'allow_large_jobs' to "
                                 "True." % stim.duration)

        if not self.validate_keys():
            raise ValueError("The provided environment key was invalid or "
                             "unauthorized. Please confirm that you have "
                             "authorized credentials for accessing the target "
                             "API.")

        return super(APITransformer, self)._transform(stim, *args, **kwargs)
