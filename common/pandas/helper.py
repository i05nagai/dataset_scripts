import pandas as pd


def read_csv(path):
    return pd.read_csv(path)


def filter_row_key_regex(df, key, regex):
    return df[df[key].fillna(value='False').str.contains(regex)]
