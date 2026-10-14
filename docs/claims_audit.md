# Claim audit: Plated in Gold app and README

Every sentence with a number in the app is generated in `app.js` from `window.STORY_DATA` (`app/data.js`). `app/build_data.py` builds that file by calling `src.story.compute_story()`, which uses `src.data.load_clean()` (cleaning in `src/cleaning.py`, features in `src/features.py`) and `src.modeling` (CatBoost settings from the original CatBoost notebook). The README and `notebooks/01_plated_in_gold_analysis.ipynb` quote the same values. Keys below refer to the `compute_story()` output: `summary.*`, `claims.*`, `gold.*`, `listing.*`, `uk.*`, `metrics*`, `importance*`, `s2gold`.

**Section order (README, notebook and app):** Intro & question → Data at a glance → Gold-plating pitfall → Modeling (CatBoost) → SHAP (global ranking, two listings explained, results by feature / UK effect) → Key findings → Limitations → Future work.

Status: ✅ supported as written · ✏️ rewritten to match data/model · ❌ removed.

| # | Section | Claim (as shown in app) | Value in app | Source (build_data.py) | Status |
|---|---|---|---|---|---|
| 1 | Intro | "We analyze 2,328 … listings" (README: "thousands") | 2,328 | `summary.entries` = len(`src.data.load_clean()`) | ✏️ |
| 2 | Intro | "factors most associated with list price" (was "impact") | — | wording; listing data is observational | ✏️ |
| 3 | Intro | regex extraction of model, Hermès, titanium, ceramic, 24k gold | — | `src/feature_engineering.add_premium_cols` patterns | ✅ |
| 4 | Intro | list prices, not transactions | — | CSV has `price` only (`claims.csv_columns`) | ✅ |
| 5 | SHAP: ranking | model 1st, then country, condition, seller ID (both variants) | model, country, condition, seller_id | `claims.model_rank_A`, `claims.model_rank_B` | ✅ |
| 6 | SHAP: ranking | model moves prediction ±$27 | 26.94 (A) / 27.03 (B) | `importance[0].value` | ✅ |
| 7 | Ranking caption | mean \|SHAP\| over held-out test set | 397 rows | `src.modeling.fit_catboost` → `FitResult.importance()` | ✏️ (was "every listing") |
| 8 | Gold pitfall | Series 2, 4, 3 average more than Ultra 3 | $743 / $676 / $664 vs $651, Ultra 3 rank 4/18 | `claims.old_above_ultra3`, `claims.model_means`, `claims.ultra3_rank` | ✅ verified |
| 9 | Gold pitfall | hiding gold → Ultra 3 rank 1 | 1 | `claims.ultra3_rank_std` (gold==0 means) | ✅ |
| 10 | Gold pitfall | 88% of Series 2 gold-plated | 88.0% | `claims.series2_gold_pct` | ✅ |
| 11 | Gold pitfall | gold premium +54.8% ($294) within models that have gold | 54.8% / $294, 9 models, 123 listings | `gold.pct`, `gold.dollar` via `src.features.calculate_price_increase`; `claims.gold_models_n`; `gold.n` | ✏️ (README "50%" / "150%") |
| 12 | Key finding 2 | README "≈150% increase" | — | contradicted by `gold.pct` = 54.8 | ❌ removed |
| 13 | Gold pitfall | top-3 gold-share models = top-3 priced | Series 2, 4, 3 both | `claims.top3_gold`, `claims.top3_price` | ✅ |
| 14 | Gold pitfall | 0 current-gen listings gold-plated (was "no current-gen") | 0 | `claims.current_gen_gold_n` | ✏️ stated as count |
| 15 | Gold pitfall | without gold, Series 2 & 3 still above Series 11 | $615, $575 vs $569 | `claims.std_means`, `claims.std_old_above_s11` | ✏️ (replaces "doesn't completely resolve") |
| 16 | Glance | raw rows; Apple share | 3,607; 65.7% | `summary.raw_rows`, `summary.apple_share` | ✅ |
| 17 | Glance | cleaning steps | 7 country labels mapped, >$2,000 dropped, 42 Apple rows dropped, 33 conditions changed, 6 "for parts" kept | `claims.country_translated_n`, `max_price_rule`, `apple_removed_by_cleaning`, `cond_changed_n`, `parts_kept_n`; `src/cleaning.py` | ✏️ ("translated listings" & "removed broken-for-parts lots" not in code → removed) |
| 18 | Glance | cards: entries / sellers / countries / models | 2,328 / 1,430 / 20 / 18 | `summary.*` | ✅ |
| 19 | Glance | USA share; UK next | 77.4%; 9.1% | `claims.us_share`, `claims.uk_share` | ✏️ (was "vast majority") |
| 20 | Glance | Ultra share (was "most watches are Ultra") | 50.0% (Series 46.8%) | `claims.ultra_share`, `claims.series_share` | ✏️ |
| 21 | Glance | UK lowest average (30+ listings); excluding UK raises mean | $578 → $593 | `claims.uk_lowest`, `claims.mean_all`, `claims.mean_ex_uk` | ✅ |
| 22 | Glance | individuals 5.8% cheaper than power sellers (2+); p | $561 vs $596; p=5.1e-8 | `listing.*` (power_seller ≥2, Welch t-test) | ✏️ (README "10%") |
| 23 | Glance | README "about 10% cheaper" | — | contradicted by `listing.pct_cheaper` | ❌ removed |
| 24 | Glance | power_seller rank 11/12, ±$3 | 2.83 | `importance` | ✅ new |
| 25 | SHAP | method description (credit split over feature orderings) | — | SHAP definition | ✅ |
| 26 | SHAP | README "Gold & Series 2 synergy → both positive" | — | contradicted: `s2gold.shap` gold +47.59, model −25.76 | ✏️ uses model result |
| 27 | SHAP | README "Gold & Ultra 3 dissonance → both negative" | — | untestable (0 gold Ultra 3: `claims.ultra3_gold_n`) | ❌ removed |
| 28 | SHAP | listing volume: 50% Ultra, Series 2 has 25 listings | 50.0%, 25 | `claims.ultra_share`, `claims.model_n` | ✏️ |
| 29 | SHAP | "SHAP is the single best way…" / "natively handles correlated data" | — | not demonstrable from data | ❌ removed; softened to "splits credit among correlated features" |
| 30 | SHAP | 1,430 seller IDs (was "thousands"); seller ID rank 4 | 1,430; 4 | `summary.sellers`, `importance` | ✏️ |
| 31 | Explorer | base value | $577 | `base` (CatBoost expected value, 11-feature model) | ✅ |
| 32 | Explorer | Typical Ultra 3: model +$62.08, condition +$25.97, shipping −$7.69 | listing 1025 | `src.modeling.typical_ultra3_index`, `claims.typical_shap` | ✅ |
| 33 | Explorer | Series 2 gold: gold +$47.59, model −$25.76 | hypothetical | `src.modeling.series2_gold_example`, `s2gold` | ✅ |
| 34 | Modeling / SHAP | train/test 1,585/397; R² 0.53; MAE $69 | 0.5297 / $69.40 | `metrics` (11-feature) | ✅ |
| 35 | Modeling / SHAP | power_seller variant R² 0.52, MAE $70 | 0.5248 / $70.02 | `metrics_ps` | ✅ |
| 36 | Modeling / SHAP | country cluster left of zero is UK; UK mean country SHAP vs others | −$106 vs +$9; UK = 98% of the 212 lowest | `claims.uk_country_shap_mean`, `other_country_shap_mean`, `uk_share_of_lowest_country_shap` | ✅ |
| 37 | Modeling / SHAP | README "UK cheaper across all models" → UK below in 17 of 17 models it lists, avg −$127 | 17/17 | `claims.uk_models_below`, `uk_models_total`, `uk_dev_mean` | ✏️ stated as count |
| 38 | Modeling / SHAP | UK country SHAP negative: 212/212 (33/33 test) | — | `uk.neg_all`, `uk.neg_test` | ✏️ stated as count (was "all") |
| 39 | Findings 1 | model top feature; Series 2/4/3 above Ultra 3 with highest gold shares 88/51/42% | — | `importance`, `claims.*` | ✏️ ("strongest predictor" → "top feature") |
| 40 | Findings 2 | gold +54.8% but rank 7/12, mean \|SHAP\| $9 | — | `gold.pct`, `claims.gold_rank`, `gold_mean_abs_shap` | ✏️ |
| 41 | Findings 3 | country #2; UK $428 vs $593, p=1.8e-52; 17/17 models | — | `importance`, `uk.*`, `claims.*` | ✏️ ("important" → "large source of variation", with numbers) |
| 42 | Limitations | 8 CSV columns, no description/sale price | 8 | `claims.csv_columns` | ✅ |
| 43 | Limitations | 33 condition/title conflicts; 65 listings without detectable model | 33; 65 | `claims.cond_changed_n`; `summary.entries − Σ claims.model_n` | ✏️ (quantified) |
| 44 | Future | proposals labelled as proposals, not findings | — | — | ✏️ |
| 45 | Charts | all charts/heatmap/filters computed live from `listings` | — | `listings[]` | ✅ |
| 46 | README | Same claims as rows 1–45, quoted with the same values; figures in `images/` come from `scripts/make_figures.py` | — | `src.story.compute_story`, `src.visualization.fig_*` | ✅ |
| 47 | App | Section order changed to match the README (data → pitfall → modeling → SHAP); no claims added | — | `app/index.html` | ✏️ |
