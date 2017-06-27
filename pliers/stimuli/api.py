import os

from .base import load_stims
from .compound import CompoundStim
from .image import ImageStim
from .text import TextStim
from .video import VideoStim
from pliers.transformers import EnvironmentKeyMixin


class TweetStim(CompoundStim, EnvironmentKeyMixin):

    _allowed_types = (TextStim, ImageStim, VideoStim)
    _allow_multiple = True
    _primary = TextStim
    _env_keys = ('TWITTER_CONSUMER_KEY', 'TWITTER_CONSUMER_SECRET',
                 'TWITTER_ACCESS_TOKEN_KEY', 'TWITTER_ACCESS_TOKEN_SECRET')

    def __init__(self, consumer_key=None, consumer_secret=None,
                 access_token_key=None, access_token_secret=None,
                 status_id=None):
        import twitter

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
        self.status = self.api.GetStatus(status_id)
        elements = [TextStim(text=self.status.text)]
        if self.status.media:
            elements.extend(load_stims([m.url for m in self.status.media]))
        super(TweetStim, self).__init__(elements=elements)
