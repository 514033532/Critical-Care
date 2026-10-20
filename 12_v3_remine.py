"""
12_v3_remine.py — re-mine GSE216009 (IL1R2+ vs IL1R2- neutrophil DE) and
GSE167363 (NK subtype) from the existing annotated object.
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


def mwu(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    return stats.mannwhitneyu(x, y, alternative="two-sided")


def main():
    a = ad.read_h5ad(C.ANNOTATED_H5AD)
    a.var_names_make_unique()
    if not a.obs_names.is_unique:
        a.obs_names_make_unique()

    # ================= GSE216009: IL1R2+ vs IL1R2- neutrophil DE =================
    g216 = a[a.obs["dataset"] == "GSE216009"]
    neu = g216[g216.obs["cell_type"] == "Neutrophils"].copy()
    il1r2 = H.gene_col(neu, "IL1R2")
    neu.obs["il1r2_pos"] = (il1r2 > 0).astype(str)
    n_pos = int((neu.obs["il1r2_pos"] == "True").sum())
    n_neg = int((neu.obs["il1r2_pos"] == "False").sum())
    print(f"GSE216009 neutrophils: {neu.shape[0]} | IL1R2+ {n_pos} / IL1R2- {n_neg}", flush=True)

    mask_pos = (neu.obs["il1r2_pos"] == "True").values
    mask_neg = (neu.obs["il1r2_pos"] == "False").values
    X = neu.X
    pos = np.asarray(X[mask_pos].mean(axis=0)).ravel()
    neg = np.asarray(X[mask_neg].mean(axis=0)).ravel()
    genes = list(neu.var_names)
    eps = 1e-3
    logfc = np.log2((pos + eps) / (neg + eps))
    de = pd.DataFrame({"gene": genes, "mean_IL1R2pos": pos, "mean_IL1R2neg": neg, "log2FC": logfc})
    de = de.sort_values("log2FC", ascending=False)
    C.save_table(de, "v3_GSE216009_IL1R2pos_vs_neg_DE")
    print("\nTop 25 genes enriched in IL1R2+ neutrophils (by log2FC):")
    print(de.head(25)[["gene", "mean_IL1R2pos", "mean_IL1R2neg", "log2FC"]].round(3).to_string(index=False))

    # ARG1 & co-expression module specifically (with MWU p-value)
    print("\n=== 目标基因（IL1R2+ vs IL1R2-，MWU p）===")
    for g in ["ARG1", "IL1R2", "PADI4", "LGALS1", "ITGAM", "CD47", "TSPO", "OLFM4", "MME", "CD38"]:
        idx = genes.index(g) if g in genes else -1
        if idx < 0:
            print(f"  {g}: 不在基因集"); continue
        xp = X[mask_pos][:, idx].toarray().ravel()
        xn = X[mask_neg][:, idx].toarray().ravel()
        U, p = mwu(xp, xn)
        print(f"  {g:8s}: pos={pos[idx]:.3f} neg={neg[idx]:.3f} log2FC={logfc[idx]:+.2f} p={p:.2e}")

    # ================= GSE167363: NK subtype =================
    g167 = a[a.obs["dataset"] == "GSE167363"]
    nk = g167[g167.obs["cell_type"] == "NK cells"].copy()
    subtypes = {
        "CD56dim_cytotoxic": H.NK_CD56DIM,
        "CD56bright_regulatory": H.NK_CD56BRIGHT,
        "Exhausted": H.NK_EXHAUSTED,
        "Proliferating": H.NK_PROLIF,
        "Tissue_homing": H.NK_HOMING,
    }
    score_cols = []
    for nm, gs in subtypes.items():
        nk.obs[f"{nm}_score"] = H.module_score(nk, gs)
        score_cols.append(f"{nm}_score")
    arr = nk.obs[score_cols].values
    nk.obs["subtype"] = np.array(list(subtypes))[np.argmax(arr, axis=1)]
    comp = nk.obs.groupby(["group", "subtype"], observed=True).size().unstack(fill_value=0)
    comp_pct = (comp.div(comp.sum(axis=1), axis=0) * 100).round(2)
    print("\n=== GSE167363 NK 亚型组成（%）===")
    print(comp_pct.to_string())
    C.save_table(comp_pct, "v3_GSE167363_nk_subtype_composition")
    score_by_group = nk.obs.groupby("group", observed=True)[score_cols].median().round(3)
    print("\n=== GSE167363 NK 亚型评分中位（按组）===")
    print(score_by_group.to_string())
    C.save_table(score_by_group, "v3_GSE167363_nk_subtype_scores")

    # pairwise MWU: survivor/non-survivor vs healthy for key subtype scores
    print("\n=== GSE167363 亚型评分 MWU（vs Healthy）===")
    for g2 in ["Survivor", "Non-survivor"]:
        for sc_ in ["Exhausted_score", "CD56dim_cytotoxic_score", "Proliferating_score"]:
            x = nk.obs[nk.obs["group"] == "Healthy"][sc_].values
            y = nk.obs[nk.obs["group"] == g2][sc_].values
            U, p = mwu(x, y)
            print(f"  {g2} vs Healthy  {sc_}: p={p:.3g}")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
