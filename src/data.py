"""Load the raw CSV and run the cleaning + feature-engineering pipeline."""
import pandas as pd
from . import RAW_CSV
from .cleaning import clean_dataframe
from .features import add_model_col, add_family_col, add_premium_cols, add_power_seller


def load_raw(path=RAW_CSV):
    return pd.read_csv(path)


def load_clean(path=RAW_CSV):
    """Raw CSV -> cleaned Apple listings with model, family, premium and power_seller columns."""
    df = clean_dataframe(load_raw(path))
    df = add_model_col(df)
    df = add_family_col(df)
    df = add_premium_cols(df)
    return add_power_seller(df)
