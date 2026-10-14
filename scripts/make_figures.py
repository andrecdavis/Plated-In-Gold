"""Regenerate every figure in images/ from the raw CSV. Run from repo root: python scripts/make_figures.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from src import IMAGES_DIR
from src.data import load_raw, load_clean
from src.modeling import FEATURES
from src.story import compute_story, CURRENT_GEN
from src import visualization as v

IMAGES_DIR.mkdir(exist_ok=True)
P = lambda n: IMAGES_DIR / n
raw, df = load_raw(), load_clean()
st = compute_story(raw, df)
lab = [f.replace('_', ' ') for f in FEATURES]
X = st['X']

v.fig_donut(df['country'].value_counts(), 'Listings by country', P('01_country_donut.png'), top=5)
v.fig_donut(df['family'].fillna('Unknown').value_counts(), 'Listings by family', P('02_family_donut.png'))
v.fig_country_means(df, P('03_country_means.png'))
v.fig_listing_type(df, P('04_listing_type.png'))
v.fig_price_by_model(df, CURRENT_GEN, P('05_price_by_model.png'))
v.fig_gold_share(df, P('06_gold_share_by_model.png'))
v.fig_price_by_model(df[df.gold == 0], CURRENT_GEN, P('07_price_by_model_no_gold.png'), 'Average list price by model, gold-plated excluded')
v.fig_importance(st['A'].importance().rename(lambda s: s.replace('_', ' ')), 'Feature importance (CatBoost + SHAP, with power seller)', P('08_shap_ranking.png'))
ti = df.index.get_loc(st['typical_idx'])
v.fig_waterfall(st['sv'][ti], [f'{l} = {"missing" if pd.isna(X.iloc[ti][f]) else X.iloc[ti][f]}' for l, f in zip(lab, FEATURES)], st['base'], 'SHAP breakdown: typical US Ultra 3 listing', P('09_shap_typical_ultra3.png'))
w = st['s2_row'].iloc[0]
v.fig_waterfall(st['s2_shap'], [f'{l} = {"missing" if pd.isna(w[f]) else w[f]}' for l, f in zip(lab, FEATURES)], st['base'], 'SHAP breakdown: hypothetical gold-plated Series 2', P('10_shap_series2_gold.png'))
v.fig_deviation_heatmap(df, P('12_price_deviation_heatmap.png'))
print('figures written to', IMAGES_DIR)
