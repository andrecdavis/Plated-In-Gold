"""Generate notebooks/01_plated_in_gold_analysis.ipynb (unexecuted); `make notebook` executes it."""
import os
import nbformat as nbf
nb = nbf.v4.new_notebook(); C=[]
md=lambda s: C.append(nbf.v4.new_markdown_cell(s)); code=lambda s: C.append(nbf.v4.new_code_cell(s))
md("# Plated in Gold: what drives Apple Watch resale list prices?\n\n**Question:** which product, seller and geographic factors are associated with higher or lower Apple Watch asking prices on resale marketplaces? *Caveat: these are list prices, not completed sales.*")
md("## 1. Setup")
code("""import sys, pathlib
ROOT = pathlib.Path.cwd().resolve().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from src import IMAGES_DIR
from src.data import load_raw, load_clean
from src.modeling import FEATURES, FEATURES_PS
from src.story import compute_story, CURRENT_GEN
from src import visualization as viz
pd.set_option('display.max_colwidth', 80)
IMAGES_DIR.mkdir(exist_ok=True)""")
md("## 2. Data at a glance\nCleaning: accents removed and titles lower-cased, Spanish condition/country labels mapped to English, prices above $2,000 dropped, Apple only, bundles/lots/iPhones removed, condition set to *New* when the title says “brand new” or “unopened”. Features extracted from titles: model, family, Hermès, cellular, titanium, 24k gold, premium band, ceramic, power seller (2+ listings).")
code("""raw = load_raw(); df = load_clean()
st = compute_story(raw, df); C = st['claims']
print(f"raw rows: {len(raw):,} | Apple share: {st['summary']['apple_share']}%")
pd.Series(st['summary'])""")
code("df.sample(5, random_state=1)[['condition','case_size','country','price','seller_id','model','gold','hermes','power_seller','title']]")
code("""print(f"USA {C['us_share']}% | UK {C['uk_share']}% | Ultra {C['ultra_share']}% | Series {C['series_share']}%")
viz.fig_donut(df['country'].value_counts(), 'Listings by country', IMAGES_DIR/'01_country_donut.png', top=5); plt.show()
viz.fig_donut(df['family'].fillna('Unknown').value_counts(), 'Listings by family', IMAGES_DIR/'02_family_donut.png'); plt.show()""")
code("""print(f"overall mean ${C['mean_all']} | excluding UK ${C['mean_ex_uk']} | UK lowest (30+ listings): {C['uk_lowest']}")
viz.fig_country_means(df, IMAGES_DIR/'03_country_means.png'); plt.show()""")
code("""print(st['listing'])
viz.fig_listing_type(df, IMAGES_DIR/'04_listing_type.png'); plt.show()""")
md("## 3. The gold-plating pitfall")
code("""print('Models averaging more than the Ultra 3:', C['old_above_ultra3'], '| Ultra 3 rank', C['ultra3_rank'])
viz.fig_price_by_model(df, CURRENT_GEN, IMAGES_DIR/'05_price_by_model.png'); plt.show()""")
code("""print('Highest gold share:', C['top3_gold'], '| top-3 priced:', C['top3_price'], '| current-gen gold listings:', C['current_gen_gold_n'])
print('Gold premium within models (src.features.calculate_price_increase):', st['gold'])
viz.fig_gold_share(df, IMAGES_DIR/'06_gold_share_by_model.png'); plt.show()""")
code("""print('Without gold: Ultra 3 rank', C['ultra3_rank_std'], '| still above Series 11:', C['std_old_above_s11'], C['std_means'])
viz.fig_price_by_model(df[df.gold == 0], CURRENT_GEN, IMAGES_DIR/'07_price_by_model_no_gold.png', 'Average list price by model, gold-plated excluded'); plt.show()""")
md("## 4. Modeling: CatBoost\nCatBoost regressor with default hyperparameters; rows with missing values dropped; 80/20 train/test split with seed 43; model, country, condition and seller ID treated as categorical features.")
code("""pd.DataFrame({'11 features (headline)': st['metrics'], '+ power_seller': st['metrics_ps']})""")
md("## 5. Explaining the model with SHAP")
code("""imp = st['A'].importance(); print(imp.round(2))
viz.fig_importance(imp.rename(lambda s: s.replace('_',' ')), 'Feature importance (CatBoost + SHAP, with power seller)', IMAGES_DIR/'08_shap_ranking.png'); plt.show()""")
code("""X = st['X']; lab = [f.replace('_',' ') for f in FEATURES]; ti = df.index.get_loc(st['typical_idx'])
print('typical Ultra 3 listing', st['typical_idx'], C['typical_shap'])
viz.fig_waterfall(st['sv'][ti], [f'{l} = {"missing" if pd.isna(X.iloc[ti][f]) else X.iloc[ti][f]}' for l, f in zip(lab, FEATURES)], st['base'], 'SHAP breakdown: typical US Ultra 3 listing', IMAGES_DIR/'09_shap_typical_ultra3.png'); plt.show()""")
code("""w = st['s2_row'].iloc[0]
print('Series 2 gold: gold', C['s2_gold_shap'], '| model', C['s2_model_shap'])
viz.fig_waterfall(st['s2_shap'], [f'{l} = {"missing" if pd.isna(w[f]) else w[f]}' for l, f in zip(lab, FEATURES)], st['base'], 'SHAP breakdown: hypothetical gold-plated Series 2', IMAGES_DIR/'10_shap_series2_gold.png'); plt.show()""")
code("""print(st['uk']); print('UK models below cross-country mean:', C['uk_models_below'], 'of', C['uk_models_total'])
viz.fig_country_shap(df['country'].values, st['sv'][:, FEATURES.index('country')], IMAGES_DIR/'11_shap_country_uk.png'); plt.show()
viz.fig_deviation_heatmap(df, IMAGES_DIR/'12_price_deviation_heatmap.png'); plt.show()""")
md("## 6. Key findings\n1. **Model is the top feature, but raw averages mislead:** Series 2, 4 and 3 average more than the Ultra 3, and they have the highest gold-plating shares.\n2. **Big niche effect, small global effect:** gold-plating carries a ~55% within-model premium but ranks 7th of 12 by mean |SHAP|.\n3. **Country is a large source of variation:** UK listings sit below the cross-country mean in 17 of the 17 models listed there, and 212 of 212 UK listings get a negative country SHAP value.")
nb['cells']=C; nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nbf.write(nb, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks', '01_plated_in_gold_analysis.ipynb'))
