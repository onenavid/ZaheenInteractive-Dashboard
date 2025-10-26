import pandas as pd


def safe_select_numeric(df, cols):
    return df[cols].select_dtypes(include=['number'])
