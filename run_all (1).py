# -*- coding: utf-8 -*-
"""
run_all.py  --  complete, corrected pipeline for the two-cohort paper
(GBDT + TreeSHAP vs glassbox EBM; Kaggle CVD cohort + Framingham teaching cohort)

Run in Colab:   %run run_all.py          (NOT "!python", because file upload needs the notebook kernel)
Outputs go to the ./results folder: tables (CSV), figures (PNG), per-bootstrap importances (CSV).

What changed compared with paperphd1.py
  1. Framingham: median imputation is FITTED ON THE TRAINING SPLIT ONLY (no leakage).
  2. One benchmark function and ONE latency method for both cohorts
     (1,000 rows, 3 warm-up calls, median of 20 timed runs, perf_counter, same n_jobs).
  3. GBDT hyperparameters are explicit per cohort and are reused in the stability analysis.
  4. EBM importances are matched BY TERM NAME (main effects only) -> valid Spearman/Jaccard.
  5. SHAP is computed on the FULL test set; Jaccard top-3/top-5 are actually computed.
  6. Test-set bootstrap 95% CIs for ROC AUC and for AUC differences (EBM vs each model).
  7. Prints counts, versions and hyperparameters needed for the Reproducibility section.
Filter rules are kept EXACTLY as in your original code (ap_hi >= ap_lo etc.), so the paper must say
that records with ap_hi < ap_lo (primary) and sysBP < diaBP (Framingham) were removed.
"""
import os, sys, time, glob, json, platform, subprocess
import numpy as np
import pandas as pd

# ------------------------------- CONFIG ---------------------------------------
CONFIG = dict(
    SEED=42,
    B=30,                       # bootstrap iterations for stability (same for both models and cohorts)
    RUN_STABILITY={'primary': True, 'framingham': True},
    TUNE_GBDT={'primary': False, 'framingham': False},   # True = 3-fold CV grid search on TRAIN split only
    AUC_CI_RESAMPLES=1000,
    STABILITY_EBM_OUTER_BAGS=None,   # None = same EBM as in the benchmark. (e.g. 2 or 4 is much faster; then state it in the paper)
    STABILITY_EBM_INTERACTIONS=None, # None = default. 0 = main effects only (fastest; state it in the paper)
    OUT_DIR='results',          # tip: mount Google Drive and use '/content/drive/MyDrive/results' so a disconnect cannot delete progress
    PRIMARY_PATH=None,          # e.g. 'cardio_train.csv'; None = auto-find, else ask for upload
    FRAMINGHAM_PATH=None,       # e.g. 'framingham.csv'
)
GBDT_PARAMS = {   # same values as in your original code; edit here if you tune
    'primary':    dict(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=CONFIG['SEED']),
    'framingham': dict(n_estimators=100, max_depth=3, random_state=CONFIG['SEED']),
}
GBDT_GRID = dict(n_estimators=[100, 200], learning_rate=[0.05, 0.1], max_depth=[2, 3, 4])

# ------------------------------- SETUP ----------------------------------------
try:
    import shap
    from interpret.glassbox import ExplainableBoostingClassifier
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'shap', 'interpret'])
    import shap
    from interpret.glassbox import ExplainableBoostingClassifier

import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, log_loss)
from sklearn.utils import resample

SEED = CONFIG['SEED']
np.random.seed(SEED)
OUT = CONFIG['OUT_DIR']
os.makedirs(OUT, exist_ok=True)

# ------------------------------- DATA -----------------------------------------
def find_inputs():
    csvs = glob.glob('*.csv') + glob.glob('/content/*.csv')
    prim = CONFIG['PRIMARY_PATH'] or next((f for f in csvs if 'framingham' not in f.lower()), None)
    fram = CONFIG['FRAMINGHAM_PATH'] or next((f for f in csvs if 'framingham' in f.lower()), None)
    if prim is None or fram is None:
        from google.colab import files
        print('Upload the primary CVD csv and the Framingham csv (file name must contain "framingham"):')
        up = files.upload()
        prim = prim or next((f for f in up if 'framingham' not in f.lower()), None)
        fram = fram or next((f for f in up if 'framingham' in f.lower()), None)
    assert prim and fram, 'Both datasets are required.'
    return prim, fram

def load_primary(path):
    df = pd.read_csv(path, delimiter=';') if ';' in open(path).readline() else pd.read_csv(path)
    n_raw = len(df)
    df['age'] = (df['age'] / 365.25).astype(int)
    keep = ((df['ap_hi'] >= df['ap_lo']) & (df['ap_hi'] > 0) & (df['ap_hi'] <= 300) &
            (df['ap_lo'] > 0) & (df['ap_lo'] <= 200) & (df['height'] >= 100) & (df['height'] <= 200) &
            (df['weight'] >= 30) & (df['weight'] <= 150))
    d = df[keep].copy()
    if 'id' in d.columns: d = d.drop('id', axis=1)
    if 'gender' in d.columns: d['gender'] = d['gender'].map({1: 0, 2: 1})
    d = pd.get_dummies(d, columns=['cholesterol', 'gluc'], drop_first=True)
    print(f'[primary] raw N = {n_raw:,} | kept = {len(d):,} | removed = {n_raw-len(d):,} ({100*(n_raw-len(d))/n_raw:.2f}%)')
    return d, 'cardio'

def load_framingham(path):
    df = pd.read_csv(path)
    n_raw = len(df)
    keep = ((df['sysBP'] >= df['diaBP']) & (df['sysBP'] > 0) & (df['sysBP'] <= 300) &
            (df['diaBP'] > 0) & (df['diaBP'] <= 200))
    d = df[keep].copy()
    d['TenYearCHD'] = d['TenYearCHD'].astype(int)
    print(f'[framingham] raw N = {n_raw:,} | kept = {len(d):,} | removed = {n_raw-len(d):,} | '
          f'positive rate = {100*d["TenYearCHD"].mean():.1f}% | missing cells = {int(d.isna().sum().sum())}')
    return d, 'TenYearCHD'

def prepare(df, target, impute):
    X, y = df.drop(target, axis=1), df[target].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    if impute:   # fitted on the TRAINING split only
        imp = SimpleImputer(strategy='median').fit(Xtr)
        Xtr = pd.DataFrame(imp.transform(Xtr), columns=Xtr.columns, index=Xtr.index)
        Xte = pd.DataFrame(imp.transform(Xte), columns=Xte.columns, index=Xte.index)
    num = Xtr.select_dtypes(include=[np.number]).columns
    sc = StandardScaler().fit(Xtr[num])
    Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
    Xtr_s[num], Xte_s[num] = sc.transform(Xtr[num]), sc.transform(Xte[num])
    print(f'  split: N_train = {len(Xtr_s):,} | N_test = {len(Xte_s):,} | test positives = {int(yte.sum())}')
    return Xtr_s, Xte_s, ytr, yte

# ------------------------------- MODELS / METRICS -----------------------------
def tune_gbdt(Xtr, ytr, base):
    gs = GridSearchCV(GradientBoostingClassifier(random_state=SEED), GBDT_GRID, scoring='roc_auc', cv=3, n_jobs=-1)
    gs.fit(Xtr, ytr)
    print('  tuned GBDT params:', gs.best_params_)
    return dict(random_state=SEED, **gs.best_params_)

def ebm_factory(**extra):
    return lambda: ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1, **extra)

def make_models(gbdt_params):
    return {
        'Logistic Regression': LogisticRegression(random_state=SEED, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=SEED, n_estimators=100, n_jobs=-1),
        'Optimized GBDT': GradientBoostingClassifier(**gbdt_params),
        'Glassbox EBM': ebm_factory()(),
    }

def latency_ms(model, X, n_rows=1000, n_runs=20, warmup=3):
    s = X.sample(n=n_rows, replace=len(X) < n_rows, random_state=0)
    for _ in range(warmup): model.predict_proba(s)
    t = []
    for _ in range(n_runs):
        a = time.perf_counter(); model.predict_proba(s); t.append((time.perf_counter() - a) * 1000)
    return float(np.median(t))

def benchmark(Xtr, Xte, ytr, yte, gbdt_params, name):
    rows, probs = [], {}
    for m, model in make_models(gbdt_params).items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        yhat = (p >= 0.5).astype(int)
        probs[m] = p
        rows.append({'Model': m, 'Accuracy': accuracy_score(yte, yhat),
                     'Precision': precision_score(yte, yhat, zero_division=0),
                     'Recall': recall_score(yte, yhat, zero_division=0),
                     'F1-Score': f1_score(yte, yhat, zero_division=0),
                     'ROC AUC': roc_auc_score(yte, p), 'AUPRC': average_precision_score(yte, p),
                     'Brier Score': brier_score_loss(yte, p), 'Log Loss': log_loss(yte, p),
                     'Latency (ms/1k, predict only)': latency_ms(model, Xte)})
    tab = pd.DataFrame(rows)
    tab.round(4).to_csv(f'{OUT}/table_performance_{name}.csv', index=False)
    print(f'\n--- PERFORMANCE: {name} ---'); print(tab.round(4).to_string(index=False))
    return tab, probs

def auc_bootstrap_ci(yte, probs, n=CONFIG['AUC_CI_RESAMPLES'], ref='Glassbox EBM', seed=0):
    """Percentile CIs from paired bootstrap resampling of the TEST set (not DeLong)."""
    rng, y = np.random.default_rng(seed), np.asarray(yte)
    store = {k: [] for k in probs}; diff = {k: [] for k in probs if k != ref}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        a = {k: roc_auc_score(y[i], p[i]) for k, p in probs.items()}
        for k in a: store[k].append(a[k])
        for k in diff: diff[k].append(a[ref] - a[k])
    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    rows = [{'Item': f'ROC AUC {k}', 'Estimate': roc_auc_score(y, probs[k]), 'CI low': ci(v)[0], 'CI high': ci(v)[1]}
            for k, v in store.items()]
    rows += [{'Item': f'AUC diff: {ref} - {k}', 'Estimate': roc_auc_score(y, probs[ref]) - roc_auc_score(y, probs[k]),
              'CI low': ci(v)[0], 'CI high': ci(v)[1]} for k, v in diff.items()]
    return pd.DataFrame(rows)

# ------------------------------- STABILITY ------------------------------------
def shap_importance(model, X_test):
    sv = shap.TreeExplainer(model).shap_values(X_test)
    if isinstance(sv, list): sv = sv[1]
    elif np.ndim(sv) == 3: sv = sv[:, :, 1]
    return np.abs(sv).mean(axis=0)

def ebm_main_importance(ebm, cols):
    imp = dict(zip(ebm.term_names_, ebm.term_importances()))
    missing = [c for c in cols if c not in imp]
    assert not missing, f'EBM term names do not match feature names: {missing[:3]}'
    return np.array([imp[c] for c in cols], dtype=float)

def summarize(vecs, cols):
    pairs = list(combinations(range(len(vecs)), 2))
    rho = np.array([spearmanr(vecs[i], vecs[j])[0] for i, j in pairs])
    top = lambda v, k: set(np.array(cols)[np.argsort(v)[::-1][:k]])
    jac = lambda a, b: len(a & b) / len(a | b)
    out = {'pairs': len(pairs), 'rho_mean': rho.mean(), 'rho_sd': rho.std(), 'rho_all': rho}
    for k in (3, 5):
        out[f'jac{k}'] = float(np.mean([jac(top(vecs[i], k), top(vecs[j], k)) for i, j in pairs]))
    return out

def hist(rho, B, path, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(rho, bins=25, color='#2a9d8f', edgecolor='black', alpha=0.85)
    ax.axvline(rho.mean(), color='red', ls='--', lw=2, label=f'Mean rho = {rho.mean():.4f} +/- {rho.std():.4f}')
    ax.set_xlabel("Spearman's rank correlation (rho)"); ax.set_ylabel(f'Frequency ({len(rho)} iteration pairs, B = {B})')
    ax.set_title(title, fontsize=10); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)

def stability(Xtr, Xte, ytr, gbdt_params, name):
    B, cols = CONFIG['B'], list(Xtr.columns)
    extra = {}
    if CONFIG['STABILITY_EBM_OUTER_BAGS'] is not None: extra['outer_bags'] = CONFIG['STABILITY_EBM_OUTER_BAGS']
    if CONFIG['STABILITY_EBM_INTERACTIONS'] is not None: extra['interactions'] = CONFIG['STABILITY_EBM_INTERACTIONS']
    mk = ebm_factory(**extra)
    tag = f"{name}_ob{CONFIG['STABILITY_EBM_OUTER_BAGS']}_int{CONFIG['STABILITY_EBM_INTERACTIONS']}"
    cp_s, cp_e = f'{OUT}/ckpt_{tag}_treeshap.csv', f'{OUT}/ckpt_{tag}_ebm.csv'
    sv, ev = [], []
    if os.path.exists(cp_s) and os.path.exists(cp_e):          # resume from saved iterations
        a, b_ = pd.read_csv(cp_s), pd.read_csv(cp_e)
        if list(a.columns) == cols and list(b_.columns) == cols and len(a) == len(b_):
            sv, ev = a.values.tolist(), b_.values.tolist()
            print(f'  [{name}] resuming: {len(sv)} bootstrap iterations already saved')
    for b in range(len(sv), B):
        t0 = time.perf_counter()
        Xb, yb = resample(Xtr, ytr, replace=True, random_state=1000 + b)
        sv.append(shap_importance(GradientBoostingClassifier(**gbdt_params).fit(Xb, yb), Xte))
        ev.append(ebm_main_importance(mk().fit(Xb, yb), cols))
        pd.DataFrame(sv, columns=cols).to_csv(cp_s, index=False)   # save after EVERY iteration
        pd.DataFrame(ev, columns=cols).to_csv(cp_e, index=False)
        print(f'  [{name}] bootstrap {b+1}/{B} done in {(time.perf_counter()-t0)/60:.1f} min', flush=True)
    pd.DataFrame(sv, columns=cols).to_csv(f'{OUT}/importances_{name}_treeshap.csv', index=False)
    pd.DataFrame(ev, columns=cols).to_csv(f'{OUT}/importances_{name}_ebm.csv', index=False)
    rs, re_ = summarize(sv, cols), summarize(ev, cols)
    hist(rs['rho_all'], B, f'{OUT}/figure2_{name}_treeshap.png', f'TreeSHAP (GBDT), {name} cohort')
    hist(re_['rho_all'], B, f'{OUT}/figure2_{name}_ebm.png', f'EBM (main-effect terms), {name} cohort')
    rows = [{'Cohort': name, 'Framework': k, 'B': B, 'Pairs': r['pairs'],
             'Spearman rho (mean +/- SD)': f"{r['rho_mean']:.4f} +/- {r['rho_sd']:.4f}",
             'Jaccard top-3': round(r['jac3'], 4), 'Jaccard top-5': round(r['jac5'], 4)}
            for k, r in (('GBDT + TreeSHAP', rs), ('Glassbox EBM', re_))]
    return pd.DataFrame(rows)

# ------------------------------- MAIN -----------------------------------------
def main():
    print('Versions: python', platform.python_version(), '| sklearn', sklearn.__version__, '| shap', shap.__version__,
          '| interpret', __import__('interpret').__version__, '| numpy', np.__version__, '| pandas', pd.__version__)
    prim_path, fram_path = find_inputs()
    cohorts = {'primary': (load_primary(prim_path), False), 'framingham': (load_framingham(fram_path), True)}
    stab_tables, auc_tables = [], []
    for name, ((df, target), impute) in cohorts.items():
        print(f'\n================ {name.upper()} ================')
        Xtr, Xte, ytr, yte = prepare(df, target, impute)
        gp = tune_gbdt(Xtr, ytr, GBDT_PARAMS[name]) if CONFIG['TUNE_GBDT'][name] else GBDT_PARAMS[name]
        print('  GBDT params used:', gp)
        tab, probs = benchmark(Xtr, Xte, ytr, yte, gp, name)
        ci = auc_bootstrap_ci(yte, probs); ci.insert(0, 'Cohort', name); auc_tables.append(ci)
        print(f'\n--- ROC AUC 95% CIs (test-set bootstrap, {CONFIG["AUC_CI_RESAMPLES"]} resamples) ---')
        print(ci.round(4).to_string(index=False))
        if CONFIG['RUN_STABILITY'][name]:
            print(f'\nRunning stability, B = {CONFIG["B"]} ...')
            stab_tables.append(stability(Xtr, Xte, ytr, gp, name))
    if auc_tables: pd.concat(auc_tables).round(4).to_csv(f'{OUT}/table_auc_ci.csv', index=False)
    if stab_tables:
        st = pd.concat(stab_tables); st.to_csv(f'{OUT}/table4_stability.csv', index=False)
        print('\n=================== TABLE 4 (stability) ==================='); print(st.to_string(index=False))
    json.dump({'config': {k: v for k, v in CONFIG.items()}, 'gbdt_params': GBDT_PARAMS}, open(f'{OUT}/run_config.json', 'w'), indent=2)
    print(f'\nAll files saved in ./{OUT}. Send me the printed tables above.')

if __name__ == '__main__' or 'get_ipython' in globals():
    main()
