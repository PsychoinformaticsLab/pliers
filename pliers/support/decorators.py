
"""Custom decorators."""
from __future__ import absolute_import
from functools import wraps
from pliers.support.exceptions import (MissingCorpusError,
                                       MissingDependencyError)


def requires_nltk_corpus(func):
    """Wraps a function that requires an NLTK corpus. If the corpus isn't
    found, raise a :exc:`MissingCorpusError`.

    Credit: borrowed from Steve Loria's TextBlob package.
    """
    @wraps(func)
    def decorated(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LookupError as err:
            print(err)
            raise MissingCorpusError()
    return decorated


def requires_optional_dependency(dependency):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AttributeError as err:
                print(err)
                raise MissingDependencyError(dependency=dependency)
        return wrapper
    return decorator
