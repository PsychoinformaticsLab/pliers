import os

from .base import load_stims
from .compound import CompoundStim
from .image import ImageStim
from .text import TextStim
from .video import VideoStim
from pliers.transformers import EnvironmentKeyMixin

try:
    import twitter
except ImportError:
    twitter = None


class TweetStimFactory(EnvironmentKeyMixin):

    '''
    An object from which to generate TweetStims, creates an Api instance from
    the python-twitter library

    Args:
        consumer_key (str): A valid consumer key for the Twitter API
        consumer_secret (str): A valid consumer secret key for the Twitter API
        access_token_key (str): A valid access token for the Twitter API
        access_token_secret (str): A valid access token secret for the
            Twitter API

    To get these credentials, visit https://dev.twitter.com/.
    '''

    _env_keys = ('TWITTER_CONSUMER_KEY', 'TWITTER_CONSUMER_SECRET',
                 'TWITTER_ACCESS_TOKEN_KEY', 'TWITTER_ACCESS_TOKEN_SECRET')

    def __init__(self, consumer_key=None, consumer_secret=None,
                 access_token_key=None, access_token_secret=None):
        if twitter is None:
            raise ImportError("python-twitter is required to create a "
                              "TweetStimFactory, but could not be successfully"
                              " imported. Please make sure it is installed.")
        if consumer_key is None or consumer_secret is None or \
           access_token_key is None or access_token_secret is None:
            try:
                consumer_key = os.environ['TWITTER_CONSUMER_KEY']
                consumer_secret = os.environ['TWITTER_CONSUMER_SECRET']
                access_token_key = os.environ['TWITTER_ACCESS_TOKEN_KEY']
                access_token_secret = os.environ['TWITTER_ACCESS_TOKEN_SECRET']
            except KeyError:
                raise ValueError("Valid Twitter API credentials "
                                 "must be passed the first time a TweetStim "
                                 "is initialized.")

        self.api = twitter.Api(consumer_key=consumer_key,
                               consumer_secret=consumer_secret,
                               access_token_key=access_token_key,
                               access_token_secret=access_token_secret)
        self.api.VerifyCredentials()

    def get_status(self, status_id):
        status = self.api.GetStatus(status_id)
        return TweetStim(status)


class TweetStim(CompoundStim):

    '''
    Represents the text and associated media from a single tweet.

    Args:
        status (python-twitter Status object): the Status from which to
            extract information, can either be generated from the TweetFactory
            or user-provided.
    '''

    _allowed_types = (TextStim, ImageStim, VideoStim)
    _allow_multiple = True
    _primary = TextStim

    def __init__(self, status):
        self.status = status
        elements = [TextStim(text=status.text)]
        if status.media:
            media_stims = load_stims([m.media_url for m in status.media])
            elements.extend(media_stims)
        super(TweetStim, self).__init__(elements=elements)
