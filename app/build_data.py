"""Build app/data.js (window.STORY_DATA) from the src pipeline. Run from repo root: python app/build_data.py"""
import json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from src.data import load_raw, load_clean
from src.modeling import FEATURES, CAT_FEATURES
from src.story import compute_story, CURRENT_GEN

raw, df = load_raw(), load_clean()
st = compute_story(raw, df)
sv, base, X = st['sv'], st['base'], st['X']
r2 = lambda x: round(float(x), 2)

w = st['s2_row']
s2gold = dict(values={k: (str(v) if k in CAT_FEATURES else (None if pd.isna(v) else float(v))) for k, v in w.iloc[0].items()},
              shap=[r2(v) for v in st['s2_shap']], pred=r2(base + st['s2_shap'].sum()))

listings = []
for j, (i, r) in enumerate(df.iterrows()):
    listings.append(dict(t=str(r['title'])[:110], m=r['model'] if pd.notna(r['model']) else 'Unknown', f=r['family'] if isinstance(r['family'], str) else 'Unknown',
        c=r['condition'] if pd.notna(r['condition']) else 'Missing', k=r['country'], p=r2(r['price']),
        s=None if pd.isna(r['case_size']) else int(r['case_size']), g=int(r['gold']), h=int(r['hermes']),
        ps=int(r['power_seller']), cel=int(r['cellular']), w=int(r['worldwide_shipping']), sid=r['seller_id'],
        pred=r2(base + sv[j].sum()), shap=[r2(v) for v in sv[j]], te=int(st['test_mask'][j]), typ=int(i == st['typical_idx'])))

out = {k: st[k] for k in ['summary', 'gold', 'hermes', 'listing', 'uk', 'metrics', 'metrics_ps', 'importance', 'importance_nops', 'claims']}
out.update(features=FEATURES, base=r2(base), s2gold=s2gold, listings=listings, current_gen=CURRENT_GEN)
with open(os.path.join(HERE, 'data.js'), 'w') as fh:
    fh.write('window.STORY_DATA = ' + json.dumps(out, separators=(',', ':')) + ';\n')
print(f"data.js written: R2={out['metrics']['r2']} MAE={out['metrics']['mae']} typical model SHAP={out['claims']['typical_shap']['model']}")
