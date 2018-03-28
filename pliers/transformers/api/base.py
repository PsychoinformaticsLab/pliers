''' Base implementation for all API transformers. '''

from pliers import config
from pliers.transformers import Transformer
from pliers.utils import isiterable, listify, APIDependent
import time


class APITransformer(APIDependent, Transformer):

    def _transform(self, stim, *args, **kwargs):
        # Check if we are requesting faster than the rate limit,
        # if so, throttle by sleeping
        time_diff = time.time() - self._last_request_time
        if time_diff < self.rate_limit:
            time.sleep(self.rate_limit - time_diff)
        self._last_request_time = time.time()

        # Check if we are trying to transform a large amount of data
        self.transformed_stim_count += len(listify(stim))
        if not config.get_option('allow_large_jobs'):
            if not isiterable(stim) and stim.duration \
               and stim.duration > config.get_option('long_job'):
                raise ValueError("Attempted to run an API transformation "
                                 "on a stimulus of duration %f, aborting. "
                                 "To allow this transformation, set "
                                 "config option 'allow_large_jobs' to "
                                 "True." % stim.duration)

            if self.transformed_stim_count > config.get_option('large_job'):
                raise ValueError("Number of transformations using this %s "
                                 "would exceed %d, aborting further "
                                 "transformations. To allow, set config "
                                 "option 'allow_large_jobs' to True." %
                                 (self.__class__.__name__,
                                  config.get_option('large_job')))

        if config.get_option('api_key_validation') and not self.validate_keys():
            raise ValueError("Error running %s, a provided environment key "
                             "was invalid or unauthorized. Please check that "
                             "you have authorized credentials for accessing "
                             "the target API." % self.__class__.__name__)

        return super(APITransformer, self)._transform(stim, *args, **kwargs)
