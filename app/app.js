const D = window.STORY_DATA, L = D.listings, F = D.features;
// Palette: Okabe-Ito (colorblind-safe). Gold is used only for gold-plated listings.
let INK = '#222222';
const BLUE = '#3A6EA5', GRAY = '#A0A4A8', ACCENT = '#E69F00', UKC = '#D55E00', GOLDHL = '#B8860B', POS = '#D55E00', NEG = '#0072B2';
const OKABE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442'];
const cfg = {responsive: true, displayModeBar: false};
// ---------- light / dark theme (light default; follows prefers-color-scheme; remembered in localStorage)
const THEMES = {light: {ink: '#222222', grid: '#E6E6E6', zero: '#BDBDBD', hoverBg: '#222222', hoverInk: '#FFFFFF'},
                dark: {ink: '#E8E8E8', grid: '#30353B', zero: '#5A6068', hoverBg: '#F2F2F2', hoverInk: '#111111'}};
let theme = localStorage.getItem('theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
const applyTheme = t => { theme = t; INK = THEMES[t].ink; document.documentElement.dataset.theme = t;
  const b = document.getElementById('themeToggle'); if (b) { b.textContent = t === 'dark' ? '☀ Light mode' : '☾ Dark mode'; b.setAttribute('aria-pressed', t === 'dark'); } };
applyTheme(theme);
const AXF = () => ({gridcolor: THEMES[theme].grid, zerolinecolor: THEMES[theme].zero, linecolor: THEMES[theme].zero, automargin: true});
const lay = (o = {}) => { const L = Object.assign({margin: {l: 70, r: 20, t: 40, b: 50}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: {family: 'Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif', size: 13, color: INK}, hoverlabel: {bgcolor: THEMES[theme].hoverBg, font: {color: THEMES[theme].hoverInk}}}, o); L.font = Object.assign({}, L.font, {color: INK}); L.xaxis = Object.assign({}, AXF(), L.xaxis); L.yaxis = Object.assign({}, AXF(), L.yaxis); return L; };
const $ = id => document.getElementById(id), fmt = v => (v < 0 ? '−$' : '$') + Math.abs(Math.round(v)).toLocaleString(), mean = a => a.reduce((s, x) => s + x, 0) / a.length;
const label = f => ({case_size: 'case size', worldwide_shipping: 'worldwide shipping', premium_band: 'premium band', power_seller: 'power seller', seller_id: 'seller id'}[f] || f);
const mkey = m => { const [a, b] = m.split(' '); return ({Series: 0, SE: 1, Ultra: 2}[a] ?? 3) * 100 + (+b || 0); };
const group = (rows, k) => rows.reduce((o, r) => ((o[r[k]] = o[r[k]] || []).push(r), o), {});
const MODELS = [...new Set(L.map(r => r.m))].filter(m => m !== 'Unknown').sort((a, b) => mkey(a) - mkey(b));
const med = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const BY_MEDIAN = [...MODELS].sort((a, b) => med(L.filter(r => r.m === a).map(r => r.p)) - med(L.filter(r => r.m === b).map(r => r.p)));

// ---------- narrative text, filled from precomputed values in data.js
const S = D.summary, C = D.claims, LS = D.listing, imp = D.importance, ci = F.indexOf('country');
const pc = v => (+v).toFixed(1) + '%', list = a => a.length < 2 ? a.join('') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1];
const H = (id, html) => { const e = $(id); if (e) e.innerHTML = html; };
H('tN', S.entries.toLocaleString());
$('rawRows').textContent = S.raw_rows.toLocaleString(); $('appleShare').textContent = pc(S.apple_share);
$('cards').innerHTML = [['Entries', S.entries], ['Sellers', S.sellers], ['Countries', S.countries], ['Models', S.models]].map(([k, v]) => `<div><b>${v.toLocaleString()}</b><small>${k}</small></div>`).join('');
$('mcards').innerHTML = `<div><b>${D.metrics.r2.toFixed(2)}</b><small>Test R²</small></div><div><b>${fmt(D.metrics.mae)}</b><small>Mean abs. error</small></div><div><b>${D.metrics.n_train}/${D.metrics.n_test}</b><small>Train / test</small></div>`;
$('baseVal').textContent = fmt(D.base);
H('tRank', `<b>${label(C.model_rank_A[0])}</b> ranks first, followed by ${list(C.model_rank_A.slice(1).map(label))} (same top four with or without the power_seller feature).`);
$('impNote').textContent = `Model moves the prediction by ±${fmt(imp[0].value)} on average.`;
H('tPit', `Grouped by model and ordered by average price, <b>${list(C.old_above_ultra3)} average more (${list(C.old_above_ultra3.map(m => fmt(C.model_means[m])))}) than the current-gen Ultra 3 (${fmt(C.ultra3_mean)}, rank ${C.ultra3_rank} of ${S.models}).</b> Is this wishful thinking, or bait-pricing? Tick <em>Hide gold-plated</em> above: the Ultra 3 moves to rank ${C.ultra3_rank_std}.`);
$('goldPct').textContent = '+' + D.gold.pct + '%';
H('goldTxt', `Many of these older listings are aftermarket <b>plated in 24k gold</b> (${C.series2_gold_pct}% of Series 2). Within the ${C.gold_models_n} models that have gold-plated listings, gold-plated watches average ${D.gold.pct}% (${fmt(D.gold.dollar)}) more than standard ones of the same model (mean of per-model gaps; ${D.gold.n} gold listings).`);
H('tGold2', `The three models with the highest gold share — ${list(C.top3_gold)} — are also the three highest-priced. ${C.current_gen_gold_n} current-gen listings are gold-plated. Gold does not explain everything: without gold-plated listings, ${list(C.std_old_above_s11)} still average more than the Series 11 (${list(C.std_old_above_s11.map(m => fmt(C.std_means[m])))} vs ${fmt(C.std_means['Series 11'])}).`);
H('tClean', `Cleaning removes accents and lower-cases titles, maps Spanish condition and country labels to English (${C.country_translated_n} country labels), drops prices above $${C.max_price_rule.toLocaleString()}, keeps Apple only, removes titles mentioning bundles, lots or iPhones (${C.apple_removed_by_cleaning} Apple rows dropped in total), and sets condition to New when the title says "brand new" or "unopened" (${C.cond_changed_n} listings changed). ${C.parts_kept_n} "for parts" listings remain.`);
H('tMix', `${pc(C.us_share)} of listings come from the USA (the UK is next at ${pc(C.uk_share)}), and among listings with an identifiable model family, ${pc(C.ultra_share)} are Ultra, Apple's flagship family (Series: ${pc(C.series_share)}).`);
H('tUKavg', `The UK has the lowest average of any country with 30+ listings. Excluding it raises the overall average from ${fmt(C.mean_all)} to ${fmt(C.mean_ex_uk)}.`);
H('listingTxt', `Watches from individuals (sellers with one listing, n=${LS.n_ind}) average <b>${fmt(LS.individual_mean)}</b>, <b>${LS.pct_cheaper.toFixed(1)}% less</b> than power sellers with 2+ listings (${fmt(LS.power_mean)}, n=${LS.n_pow}); Welch t-test p = ${LS.p.toExponential(1)}. In the model, power_seller ranks ${imp.findIndex(d => d.feature === 'power_seller') + 1} of ${imp.length} (±${fmt(imp.find(d => d.feature === 'power_seller').value)}).`);
H('tSyn', `Features interact: in a hypothetical gold-plated Series 2, <em>gold = 1</em> adds ${'$' + C.s2_gold_shap.toFixed(2)} while <em>model = Series 2</em> subtracts ${'$' + Math.abs(C.s2_model_shap).toFixed(2)}.`);
H('tPros', `<li><b>Reflects listing volume</b> — common models shape the model more; ${pc(C.ultra_share)} of listings are Ultra, while Series 2 has ${C.model_n['Series 2']} listings.</li><li><b>Splits credit among correlated features</b> — e.g. gold and older Series models.</li><li><b>Handles high-cardinality features</b> — ${C.n_sellers.toLocaleString()} seller IDs; seller ID ranks ${imp.findIndex(d => d.feature === 'seller_id') + 1} of ${imp.length}.</li>`);
H('tFit', `${D.metrics.n_train.toLocaleString()} train / ${D.metrics.n_test} test listings (80/20 split, seed 43, after dropping rows with missing values). CatBoost regressor with default hyperparameters on condition, case size, worldwide shipping, country, model, Hermès, cellular, gold, premium band, ceramic and seller ID; the ranking chart's variant also includes the power-seller feature (R² ${D.metrics_ps.r2.toFixed(2)}, MAE ${fmt(D.metrics_ps.mae)}).`);
H('tDist', `Choose a feature to see the spread of its SHAP values across all ${S.entries.toLocaleString()} listings. For <b>country</b>, the cluster left of zero is the UK: UK listings average ${fmt(C.uk_country_shap_mean)}, other countries +${fmt(C.other_country_shap_mean)}.`);
H('tUK', `UK listings average ${fmt(D.uk.uk_mean)}, against ${fmt(D.uk.rest_mean)} elsewhere (Welch p = ${D.uk.p.toExponential(1)}). This isn't just the UK listing cheaper models.`);
H('tHeat', `Each cell shows how far a country's average for a model sits from that model's average across countries (red below, blue above). <b>The UK sits below in ${C.uk_models_below} of the ${C.uk_models_total} models it lists</b>, by ${fmt(Math.abs(C.uk_dev_mean))} on average.`);
H('ukNeg', `<b>The UK effect in the model.</b> In the SHAP chart above (feature: country), UK listings sit left of zero: ${D.uk.neg_all} of ${D.uk.n} UK country SHAP values are negative (${D.uk.neg_test} of ${D.uk.n_test} in the test set) — the model lowers the prediction for being listed in the UK.`);
H('f1', `${label(C.model_rank_A[0])} has the largest mean |SHAP| (${fmt(imp[0].value)}). The Ultra 3 averages ${fmt(C.ultra3_mean)}, but ${list(C.old_above_ultra3)} average more, and they have the highest gold shares (${list(C.top3_gold.map(m => C.gold_share[m] + '%'))}).`);
H('f2', `Within models that have gold versions, gold-plating goes with a ${D.gold.pct}% higher average list price, yet gold ranks only ${C.gold_rank} of ${C.n_features} by mean |SHAP| (${fmt(C.gold_mean_abs_shap)}) — a big effect in a small niche (${D.gold.n} listings).`);
H('f3', `Country ranks #${imp.findIndex(d => d.feature === 'country') + 1}. UK listings average ${fmt(D.uk.uk_mean)} vs ${fmt(D.uk.rest_mean)} elsewhere (Welch p = ${D.uk.p.toExponential(1)}), and the UK is below the cross-country average in ${C.uk_models_below} of ${C.uk_models_total} models.`);
H('tMissing', `<b>Missing variables.</b> The CSV has only ${C.csv_columns.length} columns (${C.csv_columns.join(', ')}): no description or sale price, so product details beyond these come from the title alone.`);
H('tQuality', `<b>Web-scraped data quality.</b> Mixed-language labels and ${C.cond_changed_n} condition/title conflicts needed fixing, and regex extraction can miss or misread features (e.g. ${C.model_n ? (S.entries - Object.values(C.model_n).reduce((a, b) => a + b, 0)) : 0} listings have no detectable model).`);

const drawImp = ps => { const I = ps ? D.importance : D.importance_nops, M = ps ? D.metrics_ps : D.metrics;
  Plotly.react('impChart', [{type: 'bar', orientation: 'h', y: I.map(d => label(d.feature)).reverse(), x: I.map(d => d.value).reverse(),
  marker: {color: I.map((d, i) => i < 4 ? BLUE : GRAY).reverse()}, hovertemplate: '%{y}: ±$%{x:.2f}<extra></extra>'}],
  lay({margin: {l: 140, r: 20, t: 30, b: 40}, title: {text: `${ps ? 'With' : 'Without'} power_seller · test R² ${M.r2.toFixed(2)} · MAE ${fmt(M.mae)}`, font: {size: 13}}, xaxis: {title: 'Mean |SHAP| on test set ($)'}}), cfg); };
drawImp(true); $('impPS').onchange = e => drawImp(e.target.checked);

// ---------- filters
const fill = (id, vals) => $(id).innerHTML = '<option value="">All</option>' + vals.map(v => `<option>${v}</option>`).join('');
const byCount = k => Object.entries(group(L, k)).sort((a, b) => b[1].length - a[1].length).map(e => e[0]);
fill('fCountry', byCount('k')); fill('fFamily', ['Ultra', 'Series', 'SE']); fill('fCond', byCount('c'));
const filt = () => L.filter(r => (!$('fCountry').value || r.k === $('fCountry').value) && (!$('fFamily').value || r.f === $('fFamily').value)
  && (!$('fCond').value || r.c === $('fCond').value) && ($('fType').value === '' || r.ps === +$('fType').value) && !($('fGold').checked && r.g));
['fCountry', 'fFamily', 'fCond', 'fType', 'fGold'].forEach(id => $(id).onchange = render);
$('fReset').onclick = () => { ['fCountry', 'fFamily', 'fCond', 'fType'].forEach(id => $(id).value = ''); $('fGold').checked = false; render(); };

function render() {
  const R = filt(); $('fCount').textContent = R.length.toLocaleString() + ' listings';
  const gm = group(R.filter(r => r.m !== 'Unknown'), 'm');
  const ms = Object.keys(gm).map(m => ({m, p: mean(gm[m].map(r => r.p)), n: gm[m].length, g: mean(gm[m].map(r => r.g))})).sort((a, b) => a.p - b.p);
  Plotly.react('modelChart', [{type: 'bar', orientation: 'h', y: ms.map(d => d.m), x: ms.map(d => d.p), customdata: ms.map(d => [d.n, d.g * 100]),
    marker: {color: ms.map(d => D.current_gen.includes(d.m) ? ACCENT : BLUE)}, hovertemplate: '<b>%{y}</b><br>Avg %{x:$,.0f}<br>%{customdata[0]} listings · %{customdata[1]:.0f}% gold<extra></extra>'}],
    lay({margin: {l: 90, r: 20, t: 20, b: 40}, xaxis: {title: 'Average list price ($)', tickprefix: '$'}}), cfg);
  const gs = [...ms].sort((a, b) => a.g - b.g);
  Plotly.react('goldChart', [{type: 'bar', orientation: 'h', y: gs.map(d => d.m), x: gs.map(d => d.g * 100), customdata: gs.map(d => d.n),
    marker: {color: gs.map(d => d.g > 0 ? GOLDHL : GRAY)}, hovertemplate: '<b>%{y}</b>: %{x:.0f}% gold-plated (n=%{customdata})<extra></extra>'}],
    lay({margin: {l: 90, r: 20, t: 20, b: 40}, xaxis: {title: '% gold-plated', range: [0, 100], ticksuffix: '%'}}), cfg);
  const donut = (id, k, title, top) => { const g = Object.entries(group(R, k)).sort((a, b) => b[1].length - a[1].length);
    const head = g.slice(0, top), tail = g.slice(top), rest = tail.reduce((s, e) => s + e[1].length, 0);
    const labels = [...head.map(e => e[0] || 'Unknown'), ...(rest ? ['Other'] : [])], values = [...head.map(e => e[1].length), ...(rest ? [rest] : [])];
    const hover = [...head.map(() => ''), ...(rest ? ['<br>' + tail.map(e => `${e[0]}: ${e[1].length}`).join('<br>')] : [])];
    const tot = values.reduce((a, b) => a + b, 0);
    Plotly.react(id, [{type: 'pie', hole: .55, labels, values, customdata: hover, sort: false, direction: 'clockwise',
      marker: {colors: labels.map((l, i) => l === 'Other' || l === 'Unknown' ? GRAY : OKABE[i % OKABE.length]), line: {color: '#fff', width: 2}},
      texttemplate: values.map(v => v / tot >= .05 ? '%{percent:.1%}' : ''), textposition: 'inside', insidetextorientation: 'horizontal',
      hovertemplate: '<b>%{label}</b>: %{value} listings (%{percent:.1%})%{customdata}<extra></extra>'}],
      lay({title: {text: title, font: {size: 15}}, margin: {l: 10, r: 10, t: 44, b: 10}, showlegend: true,
        legend: {orientation: 'h', x: .5, xanchor: 'center', y: -0.02, yanchor: 'top', font: {size: 12}}}), cfg); };
  donut('countryDonut', 'k', 'Listings by country', 5); donut('modelDonut', 'f', 'Listings by family', 4);
  const gc = group(R, 'k'), cs = Object.keys(gc).filter(k => gc[k].length >= 30).map(k => ({k, p: mean(gc[k].map(r => r.p)), n: gc[k].length})).sort((a, b) => b.p - a.p), avg = R.length ? mean(R.map(r => r.p)) : 0;
  Plotly.react('countryMeans', [{type: 'bar', x: cs.map(d => d.k), y: cs.map(d => d.p), customdata: cs.map(d => d.n), marker: {color: cs.map(d => d.k === 'United Kingdom' ? UKC : BLUE)},
    hovertemplate: '<b>%{x}</b><br>Avg %{y:$,.0f} (n=%{customdata})<extra></extra>'}],
    lay({title: {text: 'Average price by country (≥30 listings)', font: {size: 15}}, yaxis: {title: 'Average list price ($)', tickprefix: '$'}, shapes: [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: avg, y1: avg, line: {dash: 'dot', color: INK}}],
      annotations: [{xref: 'paper', x: 1, y: avg, text: 'overall ' + fmt(avg), showarrow: false, yshift: 10, xanchor: 'right'}]}), cfg);
  const tr = [0, 1].map(t => ({type: 'box', name: t ? 'Power seller (2+ listings)' : 'Individual (1 listing)', y: R.filter(r => r.ps === t).map(r => r.p), boxmean: true,
    marker: {color: t ? OKABE[1] : OKABE[0]}, hovertemplate: '%{y:$,.0f}<extra></extra>'}));
  Plotly.react('listingChart', tr, lay({title: {text: 'List price by listing type', font: {size: 15}}, yaxis: {title: 'List price ($)', tickprefix: '$'}, showlegend: false}), cfg);
  const fam = ['Ultra', 'Series', 'SE'], col = {Ultra: OKABE[0], Series: OKABE[2], SE: OKABE[3]};
  // Price explorer: models on the y-axis, sorted by median price (highest at top); deterministic jitter
  const yi = Object.fromEntries(BY_MEDIAN.map((m, i) => [m, i])), jit = r => ((L.indexOf(r) * 9301 + 49297) % 233280) / 233280 - .5;
  Plotly.react('scatter', [...fam.map(f => ({rr: R.filter(r => r.f === f && r.m !== 'Unknown' && !r.g), name: f, c: col[f]})), {rr: R.filter(r => r.g && r.m !== 'Unknown'), name: 'Gold-plated', c: GOLDHL}].map(({rr, name, c}) => ({type: 'scattergl', mode: 'markers', name,
    y: rr.map(r => yi[r.m] + jit(r) * .55), x: rr.map(r => r.p), text: rr.map(r => `<b>${r.m}</b> · ${r.k} · ${r.c}${r.g ? ' · 24k gold' : ''}<br>${r.t}`),
    marker: {color: c, size: name === 'Gold-plated' ? 8 : 6, opacity: .65}, hovertemplate: '%{text}<br>%{x:$,.2f}<extra></extra>'})),
    lay({margin: {l: 80, r: 16, t: 30, b: 50}, yaxis: {tickvals: BY_MEDIAN.map((m, i) => i), ticktext: BY_MEDIAN, range: [-.7, BY_MEDIAN.length - .3], autorange: false, title: ''},
      xaxis: {title: 'List price ($)', tickprefix: '$'}, legend: {orientation: 'h', x: 0, y: 1.06}}), cfg);
  // heatmap: deviation from model mean by country (countries with <30 listings grouped as Other)
  const big = new Set(Object.keys(group(L, 'k')).filter(k => group(L, 'k')[k].length >= 30)), ck = r => big.has(r.k) ? r.k : 'Other';
  const rows = [...big, 'Other'], z = rows.map(c => MODELS.map(m => { const a = R.filter(r => ck(r) === c && r.m === m).map(r => r.p); return a.length ? a : null; }));
  const colMeans = MODELS.map((m, j) => { const v = z.map(r => r[j]).filter(Boolean).map(mean); return v.length ? mean(v) : null; });
  Plotly.react('heatmap', [{type: 'heatmap', x: MODELS, y: rows, z: z.map(r => r.map((a, j) => a ? mean(a) - colMeans[j] : null)), customdata: z.map(r => r.map(a => a ? a.length : 0)),
    colorscale: 'RdBu', reversescale: true, zmid: 0, colorbar: {title: 'Δ ($)'}, hovertemplate: '%{y} · %{x}<br>%{z:+$,.0f} vs model mean (n=%{customdata})<extra></extra>'}],
    lay({title: {text: 'Deviation from mean price by country and model', font: {size: 15}}, margin: {l: 110, r: 20, t: 40, b: 80}, xaxis: {tickangle: -45}}), cfg);
}
render();

// ---------- SHAP distributions
$('dFeat').innerHTML = imp.map(d => `<option value="${d.feature}">${label(d.feature)}</option>`).join(''); $('dFeat').value = 'country';
const dist = () => { const f = $('dFeat').value, i = F.indexOf(f), key = {country: 'k', model: 'f', condition: 'c'}[f];
  const grp = key ? group(L, key) : {all: L}, names = Object.keys(grp).sort((a, b) => grp[b].length - grp[a].length).slice(0, 6);
  Plotly.react('shapDist', names.map((n, j) => ({type: 'box', boxpoints: 'all', jitter: .6, pointpos: 0, orientation: 'h', name: n, x: grp[n].map(r => r.shap[i]), text: grp[n].map(r => r.t),
    fillcolor: 'rgba(0,0,0,0)', line: {color: 'rgba(0,0,0,0)'}, marker: {size: 4, opacity: .5, color: n === 'United Kingdom' ? UKC : [BLUE, OKABE[2], OKABE[3], OKABE[4], GRAY, OKABE[1]][j]}, hovertemplate: '%{x:+$.2f}<br>%{text}<extra></extra>'})),
    lay({title: {text: `SHAP values for '${label(f)}'`, font: {size: 15}}, margin: {l: 120, r: 20, t: 40, b: 40}, showlegend: false, xaxis: {title: 'SHAP value ($)', zeroline: true}}), cfg); };
$('dFeat').onchange = dist; dist();

// ---------- per-listing waterfall explorer
$('eModel').innerHTML = MODELS.map(m => `<option>${m}</option>`).join('');
const fillList = sel => { const rows = L.map((r, i) => [r, i]).filter(([r]) => r.m === $('eModel').value).sort((a, b) => b[0].p - a[0].p);
  $('eList').innerHTML = rows.map(([r, i]) => `<option value="${i}">${fmt(r.p)} · ${r.k} · ${r.c}${r.g ? ' · gold' : ''} — ${r.t.slice(0, 50)}</option>`).join('');
  if (sel != null) $('eList').value = sel; showListing(); };
let lastListing = null;
function showListing(r) {
  r = r || L[+$('eList').value]; if (!r) return; lastListing = r;
  const vals = {condition: r.c, case_size: r.s ? r.s + 'mm' : 'missing', cellular: r.cel ? 'Yes' : 'No', hermes: r.h ? 'Yes' : 'No', worldwide_shipping: r.w ? 'Yes' : 'No', country: r.k, model: r.m, gold: r.g ? 'Yes' : 'No', hermes: r.h ? 'Yes' : 'No', power_seller: r.ps ? 'Yes' : 'No', seller_id: r.sid};
  const o = F.map((f, i) => ({f, v: r.shap[i]})).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  Plotly.react('waterfall', [{type: 'waterfall', orientation: 'h', measure: ['absolute', ...o.map(() => 'relative'), 'total'],
    y: ['Average prediction', ...o.map(d => `${label(d.f)} = ${vals[d.f] ?? ''}`), 'Predicted price'], x: [D.base, ...o.map(d => d.v), r.pred],
    text: [fmt(D.base), ...o.map(d => (d.v >= 0 ? '+' : '−') + '$' + Math.abs(d.v).toFixed(2)), fmt(r.pred)], textposition: 'outside',
    increasing: {marker: {color: POS}}, decreasing: {marker: {color: NEG}}, totals: {marker: {color: INK}}, connector: {line: {color: '#ccc'}}, hovertemplate: '%{y}: %{text}<extra></extra>'}],
    lay({margin: {l: 230, r: 60, t: 20, b: 40}, yaxis: {autorange: 'reversed'}, xaxis: {tickprefix: '$', range: [Math.min(D.base, r.pred) - 160, Math.max(D.base, r.pred) + 160]}}), cfg);
  $('eInfo').innerHTML = `<b>${r.t}</b><br>${r.synth ? 'Hypothetical listing' : 'List price <b>' + fmt(r.p) + '</b>'} · model predicts <b>${fmt(r.pred)}</b><br>${r.k} · ${r.c} · seller ${r.sid}${r.te ? ' · <i>held-out test listing</i>' : ''}`;
}
$('eModel').onchange = () => fillList(); $('eList').onchange = showListing;
const preset = p => { let c;
  if (p === 'ultra') c = L.map((r, i) => [r, i]).filter(([r]) => r.typ);
  if (p === 's2gold') { const v = D.s2gold.values; return showListing({t: 'Hypothetical listing: a gold-plated Series 2 (other attributes taken from a real listing)', m: v.model, c: v.condition, k: v.country, p: NaN, s: v.case_size, g: 1, h: v.hermes, cel: v.cellular, ps: 0, w: v.worldwide_shipping, sid: v.seller_id, pred: D.s2gold.pred, shap: D.s2gold.shap, synth: 1}); }
  if (p === 'uk') c = L.map((r, i) => [r, i]).filter(([r]) => r.k === 'United Kingdom' && r.m === 'Ultra 2');
  if (p === 'rand') { const i = Math.floor(Math.random() * L.length); c = [[L[i], i]]; }
  if (!c || !c.length) return; const [r, i] = c[0]; if (r.m === 'Unknown') return preset('rand'); $('eModel').value = r.m; fillList(i); };
document.querySelectorAll('.presets button').forEach(b => b.onclick = () => preset(b.dataset.p)); preset('ultra');

// ---------- theme toggle: re-theme every chart
$('themeToggle').onclick = () => { applyTheme(theme === 'dark' ? 'light' : 'dark'); localStorage.setItem('theme', theme);
  render(); dist(); drawImp($('impPS').checked); if (lastListing) showListing(lastListing); };
