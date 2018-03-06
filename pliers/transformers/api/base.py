''' Base implementation for all API transformers. '''

from abc import abstractmethod, abstractproperty
from pliers import config
from pliers.transformers import Transformer
from pliers.utils import isiterable, EnvironmentKeyMixin, listify


class APITransformer(Transformer, EnvironmentKeyMixin):

    def __init__(self, **kwargs):
        self.transformed_stim_count = 0
        self.validated_keys = set()
        super(APITransformer, self).__init__(**kwargs)

    @abstractproperty
    def api_keys(self):
        pass

    def validate_keys(self):
        if all(k in self.validated_keys for k in self.api_keys):
            return True
        else:
            valid = self.check_valid_keys()
            if valid:
                for k in self.api_keys:
                    self.validated_keys.add(k)
            return valid

    @abstractmethod
    def check_valid_keys(self):
        pass

    def _transform(self, stim, *args, **kwargs):
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

        if not self.validate_keys():
            raise ValueError("Error running %s, a provided environment key "
                             "was invalid or unauthorized. Please check that "
                             "you have authorized credentials for accessing "
                             "the target API." % self.__class__.__name__)

        return super(APITransformer, self)._transform(stim, *args, **kwargs)
