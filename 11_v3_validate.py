"""
11_v3_validate.py — v3 independent-validation analyses on GSE279452 and
GSE175453 (new cohorts).  Uses the same gene-set scoring as the main pipeline.
"""
import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import common as C
import helpers as H


def load(name):
    a = ad.read_h5ad(os.path.join(C.WORK_DIR, f"v3_{name}.h5ad"))
    a.var_names_make_unique()
    if not a.obs_names.is_unique:
        a.obs_names_make_unique()
    return a


def add_scores(a):
    for nm, gs in [("NK_cytotox", H.NK_CYTOTOX), ("NK_exhaust", H.NK_EXHAUST),
                   ("NK_ifn", H.NK_IFN), ("NK_activation", H.NK_ACTIVATION)]:
        a.obs[f"{nm}_score"] = H.module_score(a, gs)
    a.obs["IL1R2"] = H.gene_col(a, "IL1R2")
    a.obs["ARG1"] = H.gene_col(a, "ARG1")


def mwu(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    return stats.mannwhitneyu(x, y, alternative="two-sided")


def nk_by_group(obs, dataset):
    nk = obs[obs["cell_type"] == "NK cells"]
    rows = []
    for g, sub in nk.groupby("group", observed=True):
        rows.append({"dataset": dataset, "group": g, "n_NK": len(sub),
                     "cytotox_median": sub["NK_cytotox_score"].median(),
                     "exhaust_median": sub["NK_exhaust_score"].median(),
                     "ifn_median": sub["NK_ifn_score"].median(),
                     "activation_median": sub["NK_activation_score"].median()})
    return pd.DataFrame(rows)


def neu_il1r2_arg1(obs, dataset):
    neu = obs[obs["cell_type"] == "Neutrophils"]
    rows = []
    for g, sub in neu.groupby("group", observed=True):
        rows.append({"dataset": dataset, "group": g, "n_neutro": len(sub),
                     "IL1R2_pos_pct": 100 * (sub["IL1R2"] > 0).mean(),
                     "ARG1_pos_pct": 100 * (sub["ARG1"] > 0).mean(),
                     "ARG1_mean": sub["ARG1"].mean(),
                     "IL1R2_mean": sub["IL1R2"].mean()})
    return pd.DataFrame(rows)


def main():
    C.setup_style()

    # ---------- GSE279452 ----------
    a = load("GSE279452")
    add_scores(a)
    obs = a.obs
    nk279 = nk_by_group(obs, "GSE279452")
    neu279 = neu_il1r2_arg1(obs, "GSE279452")
    C.save_table(nk279, "v3_GSE279452_nk_by_group")
    C.save_table(neu279, "v3_GSE279452_neutro_il1r2_arg1")
    print("=== GSE279452 NK 评分（按组中位）===")
    print(nk279.round(3).to_string(index=False))
    print("\n=== GSE279452 中性粒 IL1R2/ARG1（按组）===")
    print(neu279.round(3).to_string(index=False))

    nk = obs[obs["cell_type"] == "NK cells"]
    print("\n=== GSE279452 两两比较（MWU）===")
    for score in ["NK_exhaust_score", "NK_cytotox_score", "NK_activation_score"]:
        for g1, g2 in [("Adult_Healthy", "Adult_Sepsis"), ("Adult_ICU", "Adult_Sepsis"),
                       ("Pediatric_Healthy", "Pediatric_Sepsis"),
                       ("Pediatric_ICU", "Pediatric_Sepsis")]:
            x = nk[nk["group"] == g1][score].values
            y = nk[nk["group"] == g2][score].values
            U, p = mwu(x, y)
            print(f"  {g2} vs {g1}  {score}: n={len(x)}/{len(y)} p={p:.3g}")

    # ---------- GSE175453 ----------
    a2 = load("GSE175453")
    add_scores(a2)
    obs2 = a2.obs
    nk175 = nk_by_group(obs2, "GSE175453")
    neu175 = neu_il1r2_arg1(obs2, "GSE175453")
    C.save_table(nk175, "v3_GSE175453_nk_by_group")
    C.save_table(neu175, "v3_GSE175453_neutro_il1r2_arg1")
    print("\n=== GSE175453 NK 评分（按组中位）===")
    print(nk175.round(3).to_string(index=False))
    print("\n=== GSE175453 中性粒 IL1R2/ARG1（按组）===")
    print(neu175.round(3).to_string(index=False))

    nk2 = obs2[obs2["cell_type"] == "NK cells"]
    print("\n=== GSE175453 两两比较（MWU）===")
    for score in ["NK_exhaust_score", "NK_cytotox_score", "NK_activation_score"]:
        for g2 in ["Sepsis_CCI", "Sepsis_RR"]:
            x = nk2[nk2["group"] == "Healthy"][score].values
            y = nk2[nk2["group"] == g2][score].values
            U, p = mwu(x, y)
            print(f"  {g2} vs Healthy  {score}: n={len(x)}/{len(y)} p={p:.3g}")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
