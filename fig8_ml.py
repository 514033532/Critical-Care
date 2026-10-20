"""
Fig 8 — Machine-learning prognostic/sepsis classifier.

Sample-level features (NK proportion, NK function scores, CD38/PD-1,
mTORC1/AMPK ratio, CD4/CD8 exhaustion, monocyte ARG1) are used to build:
  (1) a sepsis-vs-healthy classifier trained on GSE290679 (n=27) and
      externally validated on GSE216009 (HV vs sepsis);
  (2) a mortality classifier trained on GSE167363 (survivor vs non-survivor,
      n=5) with leave-one-out CV (reported as a small pilot).

Outputs:
  fig8_ml.pdf / .png
  tables/fig8_sample_features.csv
  tables/fig8_ml_metrics.csv
  tables/fig8_feature_importance.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import helpers as H


def gene_col(adata, gene):
    names = list(adata.var_names)
    if gene not in names:
        return np.zeros(adata.n_obs)
    col = adata.X[:, names.index(gene)]
    return np.asarray(col.toarray()).ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()


def sample_features(adata):
    adata.obs["_stage"] = [H.map_stage(d, g) for d, g in
                           zip(adata.obs["dataset"], adata.obs["group"])]
    cyto = H.module_score(adata, H.present_genes(adata, H.NK_CYTOTOX))
    exhaust = H.module_score(adata, H.present_genes(adata, H.NK_EXHAUST))
    ifn = H.module_score(adata, H.present_genes(adata, H.NK_IFN))
    activ = H.module_score(adata, H.present_genes(adata, H.NK_ACTIVATION))
    mtor = H.module_score(adata, H.present_genes(adata, H.MTORC1))
    ampk = H.module_score(adata, H.present_genes(adata, H.AMPK))
    cd38 = gene_col(adata, "CD38")
    pdcd1 = gene_col(adata, "PDCD1")
    arg1 = gene_col(adata, "ARG1")
    ratio = (mtor + 1e-6) / (ampk + 1e-6)

    nk = H.nk_mask(adata)
    cd4 = (adata.obs["cell_type"] == "CD4 T cells").values
    cd8 = (adata.obs["cell_type"] == "CD8 T cells").values
    mono = (adata.obs["cell_type"] == "Monocytes").values

    rows = []
    for (ds, samp) in adata.obs.groupby(["dataset", "sample"]).size().index:
        m = (adata.obs["dataset"].values == ds) & (adata.obs["sample"].values == samp)
        n_total = m.sum()
        if n_total < 50:
            continue
        nk_m = m & nk; cd4_m = m & cd4; cd8_m = m & cd8; mono_m = m & mono

        def med(arr, mask):
            return float(np.median(arr[mask])) if mask.sum() >= 5 else np.nan

        rows.append({
            "dataset": ds, "sample": samp,
            "group": adata.obs.loc[m, "group"].iloc[0],
            "stage": adata.obs.loc[m, "_stage"].iloc[0],
            "n_cells": int(n_total),
            "NK_frac": float(nk_m.sum() / n_total),
            "NK_cytotoxicity": med(cyto, nk_m),
            "NK_exhaustion": med(exhaust, nk_m),
            "NK_IFN": med(ifn, nk_m),
            "NK_activation": med(activ, nk_m),
            "NK_CD38": med(cd38, nk_m),
            "NK_PDCD1": med(pdcd1, nk_m),
            "NK_mTORC1_AMPK": med(ratio, nk_m),
            "CD4_exhaustion": med(exhaust, cd4_m),
            "CD8_exhaustion": med(exhaust, cd8_m),
            "Mono_ARG1": med(arg1, mono_m),
        })
    return pd.DataFrame(rows)


FEATURES = ["NK_frac", "NK_cytotoxicity", "NK_exhaustion", "NK_IFN", "NK_activation",
            "NK_CD38", "NK_PDCD1", "NK_mTORC1_AMPK", "CD4_exhaustion", "CD8_exhaustion",
            "Mono_ARG1"]


def prep_xy(feat, y_col, groups, features=None):
    features = features or FEATURES
    sub = feat[feat["group"].isin(groups)].copy()
    sub = sub.dropna(subset=features)
    X = sub[features].values
    y = (sub["group"] == groups[1]).astype(int).values if len(groups) == 2 else None
    return sub, X, y


def main():
    C.setup_style()
    adata = H.load_adata()
    feat = sample_features(adata)
    C.save_table(feat, "fig8_sample_features")
    print(feat.groupby(["dataset", "group"]).size().to_string())

    # ============ Model 1: sepsis vs healthy ============
    tr, Xtr, ytr = prep_xy(feat, None, ["NHC", "CI-Sep"])
    te, Xte, yte = prep_xy(feat, None, ["Healthy", "Sepsis"])
    te = te[te["dataset"] == "GSE216009"]; Xte = te[FEATURES].dropna().values
    yte = (te["group"] == "Sepsis").astype(int).values
    ok = ~np.isnan(Xte).any(axis=1)
    Xte, yte, te = Xte[ok], yte[ok], te.iloc[ok]

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    metrics = []
    models = {}
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                             class_weight="balanced", max_iter=2000)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cv_auc = []
    for tr_i, va_i in cv.split(Xtr_s, ytr):
        clf.fit(Xtr_s[tr_i], ytr[tr_i])
        cv_auc.append(roc_auc_score(ytr[va_i], clf.predict_proba(Xtr_s[va_i])[:, 1]))
    clf.fit(Xtr_s, ytr)
    train_auc = roc_auc_score(ytr, clf.predict_proba(Xtr_s)[:, 1])
    test_auc = roc_auc_score(yte, clf.predict_proba(Xte_s)[:, 1]) if len(set(yte)) > 1 else np.nan
    models["L1_logreg"] = clf
    metrics.append({"model": "L1_logreg", "task": "sepsis_vs_healthy",
                    "train_n": len(ytr), "test_n": len(yte),
                    "cv_auc_mean": float(np.mean(cv_auc)), "train_auc": train_auc,
                    "test_auc": test_auc})

    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=2,
                                    random_state=0)
    cv_auc = []
    for tr_i, va_i in cv.split(Xtr_s, ytr):
        gb.fit(Xtr_s[tr_i], ytr[tr_i])
        cv_auc.append(roc_auc_score(ytr[va_i], gb.predict_proba(Xtr_s[va_i])[:, 1]))
    gb.fit(Xtr_s, ytr)
    train_auc = roc_auc_score(ytr, gb.predict_proba(Xtr_s)[:, 1])
    test_auc = roc_auc_score(yte, gb.predict_proba(Xte_s)[:, 1]) if len(set(yte)) > 1 else np.nan
    models["GBM"] = gb
    metrics.append({"model": "GBM", "task": "sepsis_vs_healthy",
                    "train_n": len(ytr), "test_n": len(yte),
                    "cv_auc_mean": float(np.mean(cv_auc)), "train_auc": train_auc,
                    "test_auc": test_auc})

    met_df = pd.DataFrame(metrics)
    C.save_table(met_df, "fig8_ml_metrics")

    # ============ Univariate feature associations (Sepsis vs Healthy) ============
    # Per-feature odds ratio (per SD) with 95% CI from univariate logistic regression,
    # pooled across GSE290679 (NHC vs CI-Sep) and GSE216009 (HV vs Sepsis).
    sep_mask = feat["group"].isin(["NHC", "CI-Sep", "Healthy", "Sepsis"])
    or_rows = []
    for f in FEATURES:
        d = feat[sep_mask].dropna(subset=[f])
        y = d["group"].isin(["CI-Sep", "Sepsis"]).astype(int).values
        x = d[f].values.astype(float)
        if x.std() < 1e-12 or len(set(y)) < 2:
            continue
        xz = (x - x.mean()) / x.std()
        m = sm.Logit(y, sm.add_constant(xz)).fit(disp=0)
        ci = m.conf_int()[1]
        or_rows.append({"feature": f,
                        "OR": float(np.exp(m.params[1])),
                        "ci_low": float(np.exp(ci[0])),
                        "ci_high": float(np.exp(ci[1])),
                        "p": float(m.pvalues[1])})
    or_df = pd.DataFrame(or_rows).sort_values("OR")
    C.save_table(or_df, "fig8_feature_odds_ratios")
    print(or_df.to_string())

    imp = pd.DataFrame({"feature": FEATURES, "coef": models["L1_logreg"].coef_[0]})
    imp = imp.sort_values("coef", ascending=False)
    C.save_table(imp, "fig8_feature_importance")
    print(met_df.to_string())
    print(imp.to_string())

    # ---- plot ----
    fig = plt.figure(figsize=(13.0, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.6)

    # (a) ROC curves
    ax = fig.add_subplot(gs[0, 0])
    l1_tr_auc = roc_auc_score(ytr, models["L1_logreg"].predict_proba(Xtr_s)[:, 1])
    l1_te_auc = (roc_auc_score(yte, models["L1_logreg"].predict_proba(Xte_s)[:, 1])
                 if len(set(yte)) > 1 else np.nan)
    gbm_tr_auc = roc_auc_score(ytr, models["GBM"].predict_proba(Xtr_s)[:, 1])
    gbm_te_auc = (roc_auc_score(yte, models["GBM"].predict_proba(Xte_s)[:, 1])
                  if len(set(yte)) > 1 else np.nan)
    fpr, tpr, _ = roc_curve(ytr, models["L1_logreg"].predict_proba(Xtr_s)[:, 1])
    ax.plot(fpr, tpr, color="#0072B2", lw=1.6,
            label=f"L1 train (AUC={l1_tr_auc:.2f})")
    if len(set(yte)) > 1:
        fpr, tpr, _ = roc_curve(yte, models["L1_logreg"].predict_proba(Xte_s)[:, 1])
        ax.plot(fpr, tpr, color="#D55E00", lw=1.6,
                label=f"L1 test (AUC={l1_te_auc:.2f})")
    fpr, tpr, _ = roc_curve(ytr, models["GBM"].predict_proba(Xtr_s)[:, 1])
    ax.plot(fpr, tpr, color="#009E73", lw=1.6, ls="--",
            label=f"GBM train (AUC={gbm_tr_auc:.2f})")
    if len(set(yte)) > 1:
        fpr, tpr, _ = roc_curve(yte, models["GBM"].predict_proba(Xte_s)[:, 1])
        ax.plot(fpr, tpr, color="#009E73", lw=1.6, ls=":",
                label=f"GBM test (AUC={gbm_te_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.9, ls="-", alpha=0.6)
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("Sepsis vs healthy classifier")
    ax.legend(fontsize=8, frameon=True, facecolor="white", framealpha=0.9,
              loc="lower right", borderpad=0.4, labelspacing=0.3, handlelength=1.0)

    # (b) feature importance
    ax = fig.add_subplot(gs[0, 1])
    colors = ["#D55E00" if v > 0 else "#0072B2" for v in imp["coef"].values]
    ax.barh(range(len(imp)), imp["coef"].values, color=colors)
    ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp["feature"].values, fontsize=9)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("L1 coefficient")
    ax.set_title("Feature contribution")

    # (c) feature odds ratios (Sepsis vs Healthy)
    ax = fig.add_subplot(gs[0, 2])
    ypos = np.arange(len(or_df))
    OR_MIN, OR_MAX = 0.05, 25.0
    or_v = np.clip(or_df["OR"].values, OR_MIN, OR_MAX)
    ci_l = np.clip(or_df["ci_low"].values, OR_MIN, OR_MAX)
    ci_h = np.clip(or_df["ci_high"].values, OR_MIN, OR_MAX)
    log_or = np.log(or_v)
    xerr_low = log_or - np.log(ci_l)
    xerr_high = np.log(ci_h) - log_or
    for i in range(len(or_df)):
        ax.errorbar(log_or[i], i, xerr=[[xerr_low[i]], [xerr_high[i]]],
                    fmt="o", markersize=4, capsize=2, lw=1.2,
                    color="#D55E00" if or_v[i] > 1 else "#0072B2")
    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_yticks(ypos); ax.set_yticklabels(or_df["feature"].values, fontsize=8)
    ax.set_xticks([np.log(0.1), np.log(0.3), np.log(1), np.log(3), np.log(10), np.log(25)])
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "25"])
    ax.set_xlim(np.log(OR_MIN), np.log(OR_MAX))
    ax.set_xlabel("Odds ratio (per SD, log scale)")
    ax.set_title("Feature association (Sepsis vs Healthy)")

    C.label_panels(fig.axes, fontsize=14)
    fig.tight_layout(rect=[0.02, 0, 1, 0.98])
    C.save_fig(fig, "fig8_ml")

    print("Fig8 done.")


if __name__ == "__main__":
    main()