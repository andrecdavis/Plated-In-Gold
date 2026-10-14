"""Every number quoted in the README and the app is computed here."""
import numpy as np
import pandas as pd
from scipy import stats
from .features import calculate_price_increase
from .modeling import FEATURES, FEATURES_PS, fit_catboost, prepare_X, shap_values, typical_ultra3_index, series2_gold_example

CURRENT_GEN = ['Ultra 3', 'Series 11', 'SE 3']
r2 = lambda x: round(float(x), 2)


def compute_story(raw, df):
    A = fit_catboost(df, FEATURES_PS)   # ranking chart variant (with power_seller)
    B = fit_catboost(df, FEATURES)      # headline metrics + explainer examples
    X = prepare_X(df, FEATURES)
    sv, base = shap_values(B.model, X)
    F = FEATURES
    ci = F.index('country')
    ukm = (df.country == 'United Kingdom').values
    test_mask = df.index.isin(B.X_test.index)

    typ = typical_ultra3_index(df, X)
    w = series2_gold_example(df, X)
    s2, _ = shap_values(B.model, w)
    s2 = s2[0]

    impA, impB = A.importance(), B.importance()
    imp = lambda s: [dict(feature=k, value=r2(v)) for k, v in s.items()]

    gd, gp = calculate_price_increase(df, 'gold')
    hd, hp = calculate_price_increase(df, 'hermes')
    ind, pw = df.loc[df.power_seller == 0, 'price'], df.loc[df.power_seller == 1, 'price']
    _, p_ls = stats.ttest_ind(ind, pw, equal_var=False)
    uk, rest = df.loc[ukm, 'price'], df.loc[~ukm, 'price']
    _, p_uk = stats.ttest_ind(uk, rest, equal_var=False)

    mm = df.groupby('model')['price'].mean().sort_values(ascending=False)
    mm_std = df[df.gold == 0].groupby('model')['price'].mean().sort_values(ascending=False)
    gshare = df.groupby('model')['gold'].mean().sort_values(ascending=False)
    fam = df['family'].value_counts(normalize=True)
    cty = df['country'].value_counts(normalize=True)
    cnt = df['country'].value_counts()
    big = cnt[cnt >= 30].index
    ck = df['country'].where(df['country'].isin(big), 'Other')
    cm = df.groupby([ck, 'model'])['price'].mean().unstack()
    dev = cm.sub(cm.mean(axis=0), axis=1).loc['United Kingdom'].dropna()
    apple = raw[raw.brand == 'Apple']

    claims = dict(
        n_listings=len(df), raw_rows=len(raw),
        model_rank_A=list(impA.index[:4]), model_rank_B=list(impB.index[:4]),
        model_means={k: r2(v) for k, v in mm.items()}, model_n={k: int(v) for k, v in df.model.value_counts().items()},
        ultra3_mean=r2(mm['Ultra 3']), ultra3_rank=int(list(mm.index).index('Ultra 3') + 1),
        old_above_ultra3=[m for m in mm.index if m not in CURRENT_GEN and mm[m] > mm['Ultra 3']],
        ultra3_rank_std=int(list(mm_std.index).index('Ultra 3') + 1),
        top3_price=list(mm.index[:3]), top3_gold=list(gshare.index[:3]),
        gold_share={k: r2(v * 100) for k, v in gshare.items()},
        current_gen_gold_n=int(df[df.model.isin(CURRENT_GEN)].gold.sum()),
        ultra3_gold_n=int(df[df.model == 'Ultra 3'].gold.sum()), series2_gold_pct=r2(gshare['Series 2'] * 100),
        gold_models_n=int((gshare > 0).sum()),
        us_share=r2(cty['United States'] * 100), uk_share=r2(cty['United Kingdom'] * 100),
        ultra_share=r2(fam.get('Ultra', 0) * 100), series_share=r2(fam.get('Series', 0) * 100),
        mean_all=r2(df.price.mean()), mean_ex_uk=r2(rest.mean()),
        uk_lowest=bool(df[df.country.isin(big)].groupby('country').price.mean().idxmin() == 'United Kingdom'),
        uk_models_below=int((dev < 0).sum()), uk_models_total=int(len(dev)), uk_dev_mean=r2(dev.mean()),
        uk_country_shap_mean=r2(sv[ukm, ci].mean()), other_country_shap_mean=r2(sv[~ukm, ci].mean()),
        uk_share_of_lowest_country_shap=r2(ukm[np.argsort(sv[:, ci])[:int(ukm.sum())]].mean() * 100),
        gold_rank=int(list(impA.index).index('gold') + 1), n_features=len(impA), gold_mean_abs_shap=r2(impA['gold']),
        n_sellers=int(df.seller_id.nunique()),
        cond_changed_n=int((df['condition'] != raw.loc[df.index, 'condition'].replace(
            {'Nuevo': 'New', 'Usado': 'Used', 'Caja abierta': 'Open box', 'New other': 'New', '--': 'Unknown'})).sum()),
        country_translated_n=int(raw.loc[df.index, 'Country'].isin(['Corea del Sur', 'Reino Unido', 'Estados Unidos']).sum()),
        apple_removed_by_cleaning=int(len(apple) - len(df)),
        parts_kept_n=int((df['condition'] == 'For parts or not working').sum()),
        std_means={m: r2(mm_std[m]) for m in ['Series 2', 'Series 3', 'Series 4', 'Ultra 3', 'Series 11'] if m in mm_std},
        std_old_above_s11=[m for m in ['Series 2', 'Series 3', 'Series 4'] if mm_std.get(m, 0) > mm_std['Series 11']],
        s2_gold_shap=r2(s2[F.index('gold')]), s2_model_shap=r2(s2[F.index('model')]),
        typical_shap={f: r2(v) for f, v in zip(F, sv[df.index.get_loc(typ)])},
        max_price_rule=2000, csv_columns=list(raw.columns),
    )
    return dict(
        A=A, B=B, X=X, sv=sv, base=base, typical_idx=typ, s2_row=w, s2_shap=s2, test_mask=test_mask,
        summary=dict(entries=len(df), sellers=int(df.seller_id.nunique()), countries=int(df.country.nunique()),
                     models=int(df.model.nunique()), apple_share=r2((raw.brand == 'Apple').mean() * 100), raw_rows=len(raw)),
        gold=dict(dollar=gd, pct=float(gp), n=int(df.gold.sum()), with_=r2(df.loc[df.gold == 1, 'price'].mean()),
                  without=r2(df.loc[df.gold == 0, 'price'].mean())),
        hermes=dict(dollar=hd, pct=float(hp)),
        listing=dict(individual_mean=r2(ind.mean()), power_mean=r2(pw.mean()), n_ind=len(ind), n_pow=len(pw),
                     pct_cheaper=r2((1 - ind.mean() / pw.mean()) * 100), p=float(p_ls)),
        uk=dict(uk_mean=r2(uk.mean()), rest_mean=r2(rest.mean()), n=int(ukm.sum()), p=float(p_uk),
                neg_all=int((sv[ukm, ci] < 0).sum()), neg_test=int((sv[test_mask & ukm, ci] < 0).sum()),
                n_test=int((test_mask & ukm).sum())),
        metrics=dict(r2=round(B.r2, 4), mae=r2(B.mae), n_train=len(B.X_train), n_test=len(B.X_test)),
        metrics_ps=dict(r2=round(A.r2, 4), mae=r2(A.mae), n_train=len(A.X_train), n_test=len(A.X_test)),
        importance=imp(impA), importance_nops=imp(impB), claims=claims,
    )
