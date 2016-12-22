# Various custom exceptions; missing corpus exception and message adapted from
# Steve Loria's TextBlob package.

MISSING_CORPUS_MESSAGE = """
One of the features you're trying to extract requires a missing nltk corpus.
To download the missing data, run:
    python -m pliers.support.download
"""


class PliersError(Exception):

    ''' Generic Pliers-related error. '''
    pass


class MissingCorpusError(PliersError):

    """Exception thrown when a user tries to use a feature that requires a
    dataset or model that the user does not have on their system.
    """

    def __init__(self, message=MISSING_CORPUS_MESSAGE, *args, **kwargs):
        super(MissingCorpusError, self).__init__(message, *args, **kwargs)
