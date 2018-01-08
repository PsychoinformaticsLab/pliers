''' Utilities to help with input/output functionality. '''

import pandas as pd


def to_long_format(df):
    ''' Convert from wide to long format, making each row a single
    feature/value pair.
    Args:
        df (DataFrame): a pandas DF that is currently in wide format (i.e.,
            each variable is in a separate column).
    Returns:
        A pandas DataFrame in long format (i.e., with columns for 'feature'
            and 'value', and all columns in the input concatenated along the
            row axis).
    '''

    # TODO: avoids circular import--should clean this up
    from pliers.extractors import ExtractorResult

    if isinstance(df, ExtractorResult):
        df = df.to_df()

    if isinstance(df.columns, pd.core.index.MultiIndex):
        ids = list(filter(lambda x: x[1] is '', df.columns))
        variables = ['extractor', 'feature']
    else:
        df = df.reset_index() if not isinstance(
            df.index, pd.Int64Index) else df
        ids = list(set(df.columns) & set(['stim', 'onset', 'duration']))
        variables = 'feature'

    values = list(set(df.columns) - set(ids))
    converted = pd.melt(df, id_vars=ids, value_vars=values, var_name=variables)
    converted.columns = [
        c[0] if isinstance(c, tuple) else c for c in converted.columns]
    return converted
