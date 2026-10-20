"""
Fig 4 — ARG1–NK functional correlation (core causal evidence).

Sample-level (GSE216009): IL1R2+ neutrophil ARG1 burden vs NK function.
Also a cross-dataset NK-function vs disease-stage view as supporting context.

Outputs:
  fig4_arg1_nk.pdf / .png
  tables/fig4_sample_level.csv
  tables/fig4_spearman.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import common as C
import helpers as H


def gene_col(adata, gene):
    names = list(adata.var_names)
    if gene not in names:
        return np.zeros(adata.n_obs)
    col = adata.X[:, names.index(gene)]
    return np.asarray(col.toarray()).ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()


def main():
    C.setup_style()
    adata = H.load_adata()
    g216 = adata.obs["dataset"] == "GSE216009"

    il1r2 = gene_col(adata, "IL1R2")
    arg1 = gene_col(adata, "ARG1")
    cyto = H.module_score(adata, H.present_genes(adata, H.NK_CYTOTOX))
    ifn = H.module_score(adata, H.present_genes(adata, H.NK_IFN))
    exhaust = H.module_score(adata, H.present_genes(adata, H.NK_EXHAUST))
    mtor = H.module_score(adata, H.present_genes(adata, H.MTORC1))
    ampk = H.module_score(adata, H.present_genes(adata, H.AMPK))

    adata.obs["_arg1"] = arg1
    adata.obs["_il1r2"] = il1r2
    adata.obs["_cyto"] = cyto
    adata.obs["_ifn"] = ifn
    adata.obs["_exhaust"] = exhaust
    adata.obs["_mtor"] = mtor
    adata.obs["_ampk"] = ampk

    neut = g216 & (adata.obs["cell_type"] == "Neutrophils")
    nk = g216 & (adata.obs["cell_type"] == "NK cells")

    # ---- sample-level aggregation ----
    rows = []
    for sid in adata.obs.loc[g216, "sample"].unique():
        n_mask = neut.values & (adata.obs["sample"].values == sid)
        k_mask = nk.values & (adata.obs["sample"].values == sid)
        n_n, n_k = n_mask.sum(), k_mask.sum()
        if n_n < 20 or n_k < 10:
            continue
        il1r2_neut = il1r2[n_mask]
        il1r2_pos = il1r2_neut > 0
        rows.append({
            "sample": sid,
            "group": adata.obs.loc[adata.obs["sample"] == sid, "group"].iloc[0],
            "n_neutrophils": n_n,
            "n_NK": n_k,
            "IL1R2pos_neutro_fraction": float(il1r2_pos.mean()),
            "ARG1_mean_neutro": float(arg1[n_mask].mean()),
            "ARG1_mean_IL1R2pos": float(arg1[n_mask][il1r2_pos].mean()) if il1r2_pos.any() else 0.0,
            "NK_cytotoxicity": float(np.median(cyto[k_mask])),
            "NK_IFN": float(np.median(ifn[k_mask])),
            "NK_exhaustion": float(np.median(exhaust[k_mask])),
            "NK_mTORC1_AMPK_ratio": float((np.median(mtor[k_mask]) + 1e-6) / (np.median(ampk[k_mask]) + 1e-6)),
        })
    sdf = pd.DataFrame(rows)
    C.save_table(sdf, "fig4_sample_level")
    print(sdf.head().to_string())

    # ---- Spearman correlations ----
    pairs = [
        ("ARG1_mean_IL1R2pos", "NK_cytotoxicity"),
        ("ARG1_mean_IL1R2pos", "NK_IFN"),
        ("ARG1_mean_neutro", "NK_cytotoxicity"),
        ("IL1R2pos_neutro_fraction", "NK_cytotoxicity"),
        ("IL1R2pos_neutro_fraction", "NK_IFN"),
        ("NK_mTORC1_AMPK_ratio", "NK_cytotoxicity"),
        ("NK_mTORC1_AMPK_ratio", "NK_IFN"),
    ]
    corr_rows = []
    for subset_name, sub in [("all", sdf), ("sepsis", sdf[sdf["group"] == "Sepsis"])]:
        if len(sub) < 3:
            continue
        for a, b in pairs:
            rho, p, ci = C.spearman(sub[a].values, sub[b].values)
            corr_rows.append({"subset": subset_name, "x": a, "y": b,
                              "n": len(sub), "rho": rho, "p": p,
                              "ci_low": ci[0], "ci_high": ci[1]})
    cdf = pd.DataFrame(corr_rows)
    cdf["p_adj_fdr"] = C.fdr(cdf["p"].values)
    C.save_table(cdf, "fig4_spearman")
    print(cdf.to_string())

    # ---- plot: scatter panels ----
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.5))
    scat_specs = [
        ("ARG1_mean_IL1R2pos", "NK_cytotoxicity", "ARG1 (IL1R2+ neutro)"),
        ("ARG1_mean_IL1R2pos", "NK_IFN", "ARG1 (IL1R2+ neutro)"),
        ("IL1R2pos_neutro_fraction", "NK_cytotoxicity", "IL1R2+ neutro fraction"),
        ("IL1R2pos_neutro_fraction", "NK_IFN", "IL1R2+ neutro fraction"),
        ("NK_mTORC1_AMPK_ratio", "NK_cytotoxicity", "NK mTORC1/AMPK ratio"),
        ("NK_mTORC1_AMPK_ratio", "NK_IFN", "NK mTORC1/AMPK ratio"),
    ]
    for ax, (x, y, xlab) in zip(axes.flat, scat_specs):
        xx = sdf[x].values; yy = sdf[y].values
        grp = sdf["group"].values
        for g, col in [("Healthy", C.PALETTE["Healthy"]), ("Sepsis", C.PALETTE["Sepsis"])]:
            m = grp == g
            ax.scatter(xx[m], yy[m], s=20, c=col, label=g, edgecolors="black",
                       linewidths=0.4, zorder=3)
        rho, p, _ = C.spearman(xx, yy)
        m = ~(np.isnan(xx) | np.isnan(yy))
        if m.sum() >= 3:
            z = np.polyfit(xx[m], yy[m], 1)
            xs = np.linspace(xx[m].min(), xx[m].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color="grey", lw=1.0, ls="--")
        ax.set_xlabel(xlab)
        ax.set_ylabel(y.replace("_", " "))
        ax.set_title(f"rho={rho:.2f}, p={p:.3f}")
        if ax is axes.flat[0]:
            ax.legend(fontsize=9, frameon=False)
    C.label_panels(axes, fontsize=14)
    fig.tight_layout()
    C.save_fig(fig, "fig4_arg1_nk")

    print("Fig4 done.")


if __name__ == "__main__":
    main()
