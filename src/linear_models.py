import pandas as pd
import numpy as np

import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns


# # =========================================================
# # MAIN MODEL FUNCTION
# # =========================================================

# def build_price_model(
#     df,
#     target='price',
#     include_features=None,
#     exclude_features=None,
#     log_transform=False,
#     min_count_for_category=5,
#     test_size=0.2,
#     random_state=42
# ):
#     """
#     Build interpretable regression model predicting watch price.

#     PARAMETERS
#     ----------
#     df : pandas DataFrame

#     target : str
#         Target variable (default = 'price')

#     include_features : list or None
#         Explicit list of features to use.
#         If None, uses all columns except target.

#     exclude_features : list or None
#         Features to remove.

#     log_transform : bool
#         If True:
#             model predicts log(price)

#     min_count_for_category : int
#         Rare categories grouped into 'OTHER'

#     RETURNS
#     -------
#     results : dict
#     """

#     model_df = df.copy()

#     # -----------------------------------------------------
#     # Feature selection
#     # -----------------------------------------------------

#     if include_features is None:

#         features = [c for c in model_df.columns if c != target]

#     else:

#         features = include_features.copy()

#     if exclude_features is not None:

#         features = [
#             f for f in features
#             if f not in exclude_features
#         ]

#     # -----------------------------------------------------
#     # Keep only needed cols
#     # -----------------------------------------------------

#     model_df = model_df[features + [target]].copy()

#     # -----------------------------------------------------
#     # Drop missing
#     # -----------------------------------------------------

#     model_df = model_df.dropna()

#     # -----------------------------------------------------
#     # Handle categorical variables
#     # -----------------------------------------------------

#     categorical_cols = model_df.select_dtypes(
#         include=['object']
#     ).columns.tolist()

#     # reduce sparse categories
#     for col in categorical_cols:

#         counts = model_df[col].value_counts()

#         rare = counts[counts < min_count_for_category].index

#         model_df[col] = model_df[col].replace(
#             rare,
#             'OTHER'
#         )

#     # -----------------------------------------------------
#     # Log transform target
#     # -----------------------------------------------------

#     if log_transform:

#         model_df[target] = np.log(model_df[target])

#     # -----------------------------------------------------
#     # One-hot encoding
#     # -----------------------------------------------------

#     X = pd.get_dummies(
#         model_df[features],
#         drop_first=True
#     )

#     y = model_df[target]

#     # -----------------------------------------------------
#     # Train/test split
#     # -----------------------------------------------------

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=test_size,
#         random_state=random_state
#     )

#     # -----------------------------------------------------
#     # Statsmodels OLS
#     # -----------------------------------------------------

#     X_train_const = sm.add_constant(X_train)
#     X_test_const = sm.add_constant(X_test)

#     model = sm.OLS(
#         y_train,
#         X_train_const
#     ).fit()

#     # -----------------------------------------------------
#     # Predictions
#     # -----------------------------------------------------

#     preds = model.predict(X_test_const)

#     # undo log transform for metrics
#     if log_transform:

#         preds_eval = np.exp(preds)
#         y_eval = np.exp(y_test)

#     else:

#         preds_eval = preds
#         y_eval = y_test

#     # -----------------------------------------------------
#     # Metrics
#     # -----------------------------------------------------

#     r2 = r2_score(y_eval, preds_eval)

#     mae = mean_absolute_error(
#         y_eval,
#         preds_eval
#     )

#     # -----------------------------------------------------
#     # Feature importance
#     # -----------------------------------------------------

#     coef_df = pd.DataFrame({
#         'feature': model.params.index,
#         'coefficient': model.params.values,
#         'abs_effect': np.abs(model.params.values),
#         'p_value': model.pvalues.values
#     })

#     coef_df = coef_df.sort_values(
#         'abs_effect',
#         ascending=False
#     )

#     # remove intercept
#     coef_df = coef_df[
#         coef_df['feature'] != 'const'
#     ]

#     # -----------------------------------------------------
#     # Output
#     # -----------------------------------------------------

#     results = {
#         'model': model,
#         'X_columns': X.columns.tolist(),
#         'feature_importance': coef_df,
#         'r2': r2,
#         'mae': mae,
#         'log_transform': log_transform,
#         'features_used': features
#     }
    
#     return results





















"""
build_price_model — OLS regression for Apple Watch marketplace listings.

Fixes applied vs. original:
  1. Rare-category grouping moved to AFTER train/test split (no leakage).
  2. np.log  →  np.log1p  (handles zero prices gracefully);
     inverse is np.expm1 to match.
  3. Explicit zero/negative price guard before log transform.
  4. Reference category for each categorical set deliberately, not alphabetically.
  5. get_dummies dtype=int to avoid bool/uint8 ambiguity.
  6. Boolean/binary columns excluded from get_dummies to avoid redundant encoding.
  7. add_constant applied by inserting a literal column, avoiding edge-case
     behaviour when a test column happens to be constant.
  8. Standardised coefficients added alongside raw coefficients for valid
     cross-feature importance comparison.
  9. VIF computed for every predictor so multicollinearity is visible.
 10. Low-variance column check with configurable threshold.
 11. seller_id excluded by default via DEFAULT_EXCLUDE.
 12. Per-column min_count override supported via min_count_for_category dict.
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, List, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# Columns excluded unless the caller explicitly overrides include_features.
# seller_id is a high-cardinality ID; it is not a structural price driver.
# ---------------------------------------------------------------------------
DEFAULT_EXCLUDE = {'seller_id', 'title'}

# ---------------------------------------------------------------------------
# Sensible reference levels for known categorical columns.
# 'drop_first=True' drops alphabetically; we want deliberate baselines.
# ---------------------------------------------------------------------------
DEFAULT_REFERENCE_LEVELS = {
    'condition': 'Used',
    'country':   'US',       # adjust to your most-common country
    'model':     None,       # None → use whatever pd.get_dummies drops first
    'family':    None,
}

# ---------------------------------------------------------------------------
# Binary columns that are already 0/1 or bool.
# get_dummies must NOT encode these; they are passed through as-is.
# ---------------------------------------------------------------------------
KNOWN_BINARY_COLS = {
    'worldwide_shipping', 'hermes', 'cellular',
    'titanium', 'gold', 'premium_band', 'ceramic',
}


def _group_rare_categories(
    series: pd.Series,
    min_count: int,
    fit_counts: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Replace rare levels with 'OTHER'.

    Parameters
    ----------
    series : pd.Series
        The column to transform (train or test split).
    min_count : int
        Levels with fewer occurrences than this in *fit_counts* become 'OTHER'.
    fit_counts : pd.Series or None
        Value counts computed on the TRAINING set.
        Pass None only when fitting (i.e. computing from series itself).

    Returns
    -------
    pd.Series with rare levels replaced.
    """
    if fit_counts is None:
        fit_counts = series.value_counts()

    rare = fit_counts[fit_counts < min_count].index
    return series.replace(rare, 'OTHER')


def _set_reference_level(series: pd.Series, reference: str) -> pd.Series:
    """
    Make *reference* the first category so that pd.get_dummies(drop_first=True)
    drops it — giving a deliberate, interpretable baseline.
    """
    cats = [reference] + [c for c in series.unique() if c != reference]
    return pd.Categorical(series, categories=cats)


def _add_constant_column(X: pd.DataFrame) -> pd.DataFrame:
    """
    Prepend a column of ones named 'const'.

    Safer than sm.add_constant when a test-set column is accidentally constant
    (sm.add_constant may skip insertion in that edge case).
    """
    out = X.copy()
    out.insert(0, 'const', 1.0)
    return out


def _compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each column in X.
    X must NOT contain the constant column.
    Columns with zero variance are skipped (VIF undefined).
    """
    valid = X.loc[:, X.var() > 0]
    vif_values = [
        variance_inflation_factor(valid.values, i)
        for i in range(valid.shape[1])
    ]
    return pd.DataFrame({
        'feature': valid.columns,
        'vif':     vif_values,
    }).sort_values('vif', ascending=False)


def build_price_model(
    df: pd.DataFrame,
    target: str = 'price',
    include_features: Optional[List[str]] = None,
    exclude_features: Optional[Union[set, list]] = None,
    log_transform: bool = False,
    min_count_for_category: Union[int, Dict] = 30,
    reference_levels: Optional[Dict] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    low_variance_threshold: float = 0.01,
) -> dict:
    """
    Build an interpretable OLS regression model predicting watch price.

    Parameters
    ----------
    df : pd.DataFrame

    target : str
        Target column (default 'price').

    include_features : list or None
        Explicit feature list. If None, all columns except *target* are used
        (minus DEFAULT_EXCLUDE and anything in *exclude_features*).

    exclude_features : set/list or None
        Additional columns to drop on top of DEFAULT_EXCLUDE.

    log_transform : bool
        If True the model predicts log1p(price); metrics are reported on the
        original price scale via expm1().

    min_count_for_category : int or dict
        Rare-category threshold. Pass a dict for per-column overrides, e.g.
        {'country': 50, 'model': 20}. Missing columns fall back to the
        'default' key or 30.

    reference_levels : dict or None
        Mapping {column: reference_level} for deliberate baseline categories.
        Merged with DEFAULT_REFERENCE_LEVELS; caller values take priority.

    test_size : float
        Fraction of data held out for evaluation.

    random_state : int

    low_variance_threshold : float
        Encoded columns with variance below this are dropped with a warning.

    Returns
    -------
    dict with keys:
        model, X_columns, feature_importance, vif, r2, mae,
        log_transform, features_used, dropped_low_variance
    """

    # ------------------------------------------------------------------
    # 0. Resolve configuration
    # ------------------------------------------------------------------
    all_exclude = set(DEFAULT_EXCLUDE)
    if exclude_features is not None:
        all_exclude.update(exclude_features)

    ref_levels = {**DEFAULT_REFERENCE_LEVELS}
    if reference_levels is not None:
        ref_levels.update(reference_levels)

    # Normalise min_count_for_category to a dict
    if isinstance(min_count_for_category, int):
        min_count_map: dict = {'default': min_count_for_category}
    else:
        min_count_map = {'default': 30, **min_count_for_category}

    # ------------------------------------------------------------------
    # 1. Feature selection
    # ------------------------------------------------------------------
    model_df = df.copy()

    if include_features is None:
        features = [
            c for c in model_df.columns
            if c != target and c not in all_exclude
        ]
    else:
        features = [f for f in include_features if f not in all_exclude]

    model_df = model_df[features + [target]].copy()

    # ------------------------------------------------------------------
    # 2. Drop rows with missing values
    # ------------------------------------------------------------------
    before = len(model_df)
    model_df = model_df.dropna()
    dropped_na = before - len(model_df)
    if dropped_na:
        warnings.warn(f"Dropped {dropped_na} rows containing NaN values.")

    # ------------------------------------------------------------------
    # 3. Guard: non-positive prices before log transform
    # ------------------------------------------------------------------
    if log_transform:
        non_positive = (model_df[target] <= 0).sum()
        if non_positive:
            warnings.warn(
                f"{non_positive} rows have {target} ≤ 0 and will be removed "
                f"before log transform."
            )
            model_df = model_df[model_df[target] > 0]

    # ------------------------------------------------------------------
    # 4. Log-transform target  (np.log1p, not np.log)
    #    Inverse is np.expm1 — the pair round-trips exactly.
    # ------------------------------------------------------------------
    if log_transform:
        model_df[target] = np.log1p(model_df[target])

    # ------------------------------------------------------------------
    # 5. Train / test split  — BEFORE rare-category grouping to prevent
    #    any leakage of test-set frequency information into the encoding.
    # ------------------------------------------------------------------
    X_raw = model_df[features]
    y     = model_df[target]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=test_size,
        random_state=random_state,
    )

    # ------------------------------------------------------------------
    # 6. Rare-category grouping (fit on train, apply to both splits)
    # ------------------------------------------------------------------
    categorical_cols = X_train_raw.select_dtypes(include='object').columns.tolist()

    # Cast binary string columns to int if they slipped through as object
    for col in list(categorical_cols):
        if col in KNOWN_BINARY_COLS:
            try:
                X_train_raw = X_train_raw.copy()
                X_test_raw  = X_test_raw.copy()
                X_train_raw[col] = X_train_raw[col].astype(int)
                X_test_raw[col]  = X_test_raw[col].astype(int)
                categorical_cols.remove(col)
            except (ValueError, TypeError):
                pass  # leave as categorical if cast fails

    train_fit_counts: Dict[str, pd.Series] = {}

    X_train_enc = X_train_raw.copy()
    X_test_enc  = X_test_raw.copy()

    for col in categorical_cols:
        threshold = min_count_map.get(col, min_count_map['default'])

        # Fit on training data only
        fit_counts = X_train_raw[col].value_counts()
        train_fit_counts[col] = fit_counts

        X_train_enc[col] = _group_rare_categories(
            X_train_enc[col], threshold, fit_counts
        )
        X_test_enc[col] = _group_rare_categories(
            X_test_enc[col], threshold, fit_counts
        )

    # ------------------------------------------------------------------
    # 7. Set deliberate reference levels for categorical columns
    # ------------------------------------------------------------------
    for col in categorical_cols:
        ref = ref_levels.get(col)
        if ref is not None and ref in X_train_enc[col].values:
            X_train_enc[col] = _set_reference_level(X_train_enc[col], ref)
            X_test_enc[col]  = _set_reference_level(X_test_enc[col],  ref)

    # ------------------------------------------------------------------
    # 8. One-hot encoding
    #    Binary columns (bool or int 0/1) are excluded from get_dummies
    #    because they are already correctly encoded.
    # ------------------------------------------------------------------
    binary_in_frame = [
        c for c in features
        if c in KNOWN_BINARY_COLS
        and c in X_train_enc.columns
    ]
    cols_to_dummy = [c for c in categorical_cols if c not in binary_in_frame]

    X_train = pd.get_dummies(
        X_train_enc,
        columns=cols_to_dummy,
        drop_first=True,
        dtype=int,
    )
    X_test = pd.get_dummies(
        X_test_enc,
        columns=cols_to_dummy,
        drop_first=True,
        dtype=int,
    )

    # Align test columns to training columns (handles unseen test categories)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # ------------------------------------------------------------------
    # 9. Low-variance column check
    # ------------------------------------------------------------------
    variances = X_train.var()
    low_var_cols = variances[variances < low_variance_threshold].index.tolist()
    if low_var_cols:
        warnings.warn(
            f"Dropping {len(low_var_cols)} low-variance column(s): {low_var_cols}"
        )
        X_train = X_train.drop(columns=low_var_cols)
        X_test  = X_test.drop(columns=low_var_cols)

    # ------------------------------------------------------------------
    # 10. Add constant (manual insertion avoids sm.add_constant edge cases)
    # ------------------------------------------------------------------
    X_train_const = _add_constant_column(X_train)
    X_test_const  = _add_constant_column(X_test)

    # ------------------------------------------------------------------
    # 11. Fit OLS
    # ------------------------------------------------------------------
    model = sm.OLS(y_train, X_train_const).fit()

    # ------------------------------------------------------------------
    # 12. Predictions — undo log1p for metric evaluation
    # ------------------------------------------------------------------
    preds = model.predict(X_test_const)

    if log_transform:
        preds_eval = np.expm1(preds)     # inverse of log1p
        y_eval     = np.expm1(y_test)
    else:
        preds_eval = preds
        y_eval     = y_test

    # ------------------------------------------------------------------
    # 13. Metrics  (y_true first, y_pred second — sklearn convention)
    # ------------------------------------------------------------------
    r2  = r2_score(y_eval, preds_eval)
    mae = mean_absolute_error(y_eval, preds_eval)

    # ------------------------------------------------------------------
    # 14. Feature importance — raw AND standardised coefficients
    #
    #     Raw coefficients are not comparable across features on different
    #     scales (e.g. a binary dummy vs. case_size in mm).
    #     Standardised coefficient = raw_coef * (std_X / std_y)
    #     This puts all features on a common "standard deviation" unit.
    # ------------------------------------------------------------------
    std_y = y_train.std()
    std_X = X_train.std().replace(0, np.nan)   # avoid division by zero

    params      = model.params.drop('const')
    pvalues     = model.pvalues.drop('const')
    std_coefs   = params * (std_X / std_y)

    coef_df = pd.DataFrame({
        'feature':         params.index,
        'coefficient':     params.values,
        'std_coefficient': std_coefs.values,   # use THIS for importance ranking
        'abs_std_effect':  std_coefs.abs().values,
        'p_value':         pvalues.values,
    }).sort_values('abs_std_effect', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 15. VIF (multicollinearity diagnostic)
    # ------------------------------------------------------------------
    vif_df = _compute_vif(X_train)

    # ------------------------------------------------------------------
    # 16. Package results
    # ------------------------------------------------------------------
    results = {
        'model':               model,
        'X_columns':           X_train.columns.tolist(),
        'feature_importance':  coef_df,
        'vif':                 vif_df,
        'r2':                  r2,
        'mae':                 mae,
        'log_transform':       log_transform,
        'features_used':       features,
        'dropped_low_variance': low_var_cols,
    }

    return results









