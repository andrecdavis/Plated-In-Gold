"""CatBoost price model + SHAP values, matching the original CatBoost notebook exactly:
dropna -> train_test_split(test_size=0.2, random_state=43) -> CatBoostRegressor() defaults,
categorical features model/country/condition/seller_id, SHAP via CatBoost 'ShapValues'."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CAT_FEATURES = ['model', 'country', 'condition', 'seller_id']
FEATURES = ['condition', 'case_size', 'worldwide_shipping', 'country', 'model', 'hermes', 'cellular',
            'gold', 'premium_band', 'ceramic', 'seller_id']
FEATURES_PS = FEATURES + ['power_seller']
RANDOM_STATE = 43


@dataclass
class FitResult:
    model: CatBoostRegressor
    features: list
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    test_shap: np.ndarray        # (n_test, n_features), base column removed
    r2: float
    mae: float

    def importance(self):
        """Mean |SHAP| per feature over the test set, descending."""
        return pd.Series(np.abs(self.test_shap).mean(0), index=self.features).sort_values(ascending=False)


def fit_catboost(df, features=FEATURES, random_state=RANDOM_STATE):
    dm = df.dropna()
    X_train, X_test, y_train, y_test = train_test_split(dm[features], dm[['price']], test_size=0.2,
                                                        random_state=random_state)
    m = CatBoostRegressor(allow_writing_files=False)
    m.fit(X_train, y_train, cat_features=CAT_FEATURES, verbose=0)
    sv = m.get_feature_importance(Pool(X_test, cat_features=CAT_FEATURES), type='ShapValues')[:, :-1]
    p = m.predict(X_test)
    return FitResult(m, list(features), X_train, X_test, y_test, sv, r2_score(y_test, p), mean_absolute_error(y_test, p))


def prepare_X(df, features=FEATURES):
    """All listings, missing categoricals filled with 'Missing' (as in the notebook's explainer cells)."""
    X = df[features].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].fillna('Missing').astype(str)
    return X


def shap_values(model, X):
    """Returns (shap matrix without base column, base value)."""
    sv = model.get_feature_importance(Pool(X, cat_features=CAT_FEATURES), type='ShapValues')
    return sv[:, :-1], float(sv[0, -1])


def typical_ultra3_index(df, X):
    """Notebook's 'typical' listing: US Ultra 3, no Hermès/gold/ceramic, closest to the median price."""
    mask = (X.model == 'Ultra 3') & (X.country == 'United States') & (X.hermes == 0) & (X.gold == 0) & (X.ceramic == 0)
    prices = df.loc[X[mask].index, 'price']
    return (prices - prices.median()).abs().idxmin()


def series2_gold_example(df, X):
    """Notebook's hypothetical: first listing with model='Series 2' and gold=1."""
    w = X.loc[[df.index[0]]].copy()
    w['model'] = 'Series 2'
    w['gold'] = 1
    return w
