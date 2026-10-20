"""
Fig 2 — IL1R2+ immature neutrophil identification and ARG1 specificity (GSE216009).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import helpers as H


def gene_col(adata, gene):
    names = list(adata.var_names)
    if gene not in names:
        return np.zeros(adata.n_obs)
    idx = names.index(gene)
    col = adata.X[:, idx]
    if hasattr(col, "toarray"):
        return np.asarray(col.toarray()).ravel()
    return np.asarray(col).ravel()


def main():
    C.setup_style()
    adata = H.load_adata()
    g216 = adata.obs["dataset"] == "GSE216009"
    neut = g216 & (adata.obs["cell_type"] == "Neutrophils")
    print(f"GSE216009 neutrophils: {neut.sum()}")

    fine = adata.obs.loc[neut, "cell_type_fine"]
    il1r2 = gene_col(adata, "IL1R2")
    arg1 = gene_col(adata, "ARG1")
    il1r2_neut = il1r2[neut.values]
    il1r2_pos = il1r2_neut > 0
    npos = il1r2_pos.sum()
    print(f"IL1R2+ neutrophils: {npos} ({100*npos/neut.sum():.1f}%)")

    diag = adata.obs.loc[neut, "diagnosis"].astype(str)
    df = pd.DataFrame({"il1r2": il1r2_neut, "diag": diag.values})
    df["il1r2_pos"] = (df["il1r2"] > 0).astype(int)
    prop = df.groupby("diag")["il1r2_pos"].agg(["mean", "sum", "count"]).reset_index()
    prop.columns = ["diagnosis", "il1r2_pos_prop", "il1r2_pos_n", "n_neutrophils"]
    prop["group"] = np.where(prop["diagnosis"] == "HV", "Healthy", "Sepsis")
    prop = prop.sort_values("il1r2_pos_prop", ascending=False)
    C.save_table(prop, "fig2_il1r2_proportion_by_group")

    subsets = {
        "IL1R2+ neutrophils": (neut.values & (il1r2 > 0)),
        "IL1R2- neutrophils": (neut.values & (il1r2 == 0)),
        "Classical monocytes": (g216 & (adata.obs["cell_type"] == "Monocytes") &
                                 (adata.obs["cell_type_fine"] == "Classical_monocytes")).values,
        "Non-classical monocytes": (g216 & (adata.obs["cell_type"] == "Monocytes") &
                                     (adata.obs["cell_type_fine"] == "Non-classical_monocytes")).values,
        "CD4 T cells": (g216 & (adata.obs["cell_type"] == "CD4 T cells")).values,
        "NK cells": (g216 & (adata.obs["cell_type"] == "NK cells")).values,
    }
    arg1_rows = []
    for name, m in subsets.items():
        a = arg1[m]
        arg1_rows.append({
            "subset": name, "n": int(m.sum()),
            "ARG1_mean": float(a.mean()),
            "ARG1_pct_expressed": float((a > 0).mean() * 100),
        })
    arg1_df = pd.DataFrame(arg1_rows)
    C.save_table(arg1_df, "fig2_arg1_by_celltype")

    gmdsc = H.module_score(adata, H.present_genes(adata, H.GMDSC))
    gmdsc_df = pd.DataFrame({
        "IL1R2": np.where(il1r2[neut.values] > 0, "IL1R2+", "IL1R2-"),
        "GMDSC_score": gmdsc[neut.values],
    })
    C.save_table(gmdsc_df, "fig2_gmdsc_score")

    co_genes = ["PADI4", "CD47", "LGALS1", "TSPO", "ITGAM", "IL1R2"]
    co_genes = [g for g in co_genes if g in list(adata.var_names)]
    coexp = pd.DataFrame(index=co_genes, columns=["pct_coexpressed_with_ARG1", "rho_with_ARG1"])
    arg1_pos_mask = (arg1[neut.values] > 0)
    for g in co_genes:
        gv = gene_col(adata, g)[neut.values]
        coexp.loc[g, "pct_coexpressed_with_ARG1"] = 100 * np.mean(gv[arg1_pos_mask] > 0)
        if arg1_pos_mask.sum() >= 5:
            rho, p, _ = C.spearman(arg1[neut.values][arg1_pos_mask], gv[arg1_pos_mask])
            coexp.loc[g, "rho_with_ARG1"] = rho
    C.save_table(coexp, "fig2_il1r2_correlation")

    fig = plt.figure(figsize=(12.5, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.65, wspace=0.55)

    # (A)
    ax = fig.add_subplot(gs[0, 0])
    order = ["Immature_neutrophils", "Neutrophil_progenitors", "Mature_neutrophils",
             "Degranulating_neutrophils", "Apoptosing_neutrophils"]
    order = [o for o in order if o in set(fine)]
    short = [o.replace("_neutrophils", "").replace("Neutrophil_", "") for o in order]
    data = [il1r2[neut.values][fine.values == o] for o in order]
    bp = ax.boxplot(data, tick_labels=short, showfliers=False, patch_artist=True, widths=0.6)
    for patch, o in zip(bp["boxes"], order):
        patch.set_facecolor(C.LINEAGE_PALETTE.get(o, "#BBBBBB"))
    ax.set_ylabel("IL1R2 expression")
    ax.set_title("IL1R2 by neutrophil subset")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Neutrophil subset")

    # (B)
    ax = fig.add_subplot(gs[0, 1])
    hv = prop[prop["diagnosis"] == "HV"]["il1r2_pos_prop"].values
    sep = prop[prop["diagnosis"] != "HV"]["il1r2_pos_prop"].values
    maxval = max(hv.max() if len(hv) else 0, sep.max() if len(sep) else 0)
    ax.bar([0, 1], [hv.mean() if len(hv) else 0, sep.mean() if len(sep) else 0],
           color=[C.PALETTE["Healthy"], C.PALETTE["Sepsis"]], width=0.5)
    for _, r in prop[prop["diagnosis"] != "HV"].iterrows():
        ax.scatter(1, r["il1r2_pos_prop"], s=20, c="black", zorder=3)
    for _, r in prop[prop["diagnosis"] == "HV"].iterrows():
        ax.scatter(0, r["il1r2_pos_prop"], s=20, c="black", zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Healthy", "Sepsis"])
    ax.set_ylabel("IL1R2+ neutrophil fraction")
    ax.set_title("IL1R2+ neutrophil frequency")
    ax.set_xlabel("Group")
    if len(hv) and len(sep):
        _, p = C.mwu(hv, sep)
        ax.set_ylim(0, maxval * 1.35)
        C.add_sig_annot(ax, 0, 1, maxval * 1.10, maxval * 0.06, p)

    # (C)
    ax = fig.add_subplot(gs[0, 2])
    names = list(subsets.keys())
    short_names = ["IL1R2+ neut", "IL1R2- neut", "Classical mono",
                   "Non-classical mono", "CD4 T", "NK"]
    boxdata = [arg1[subsets[n]] for n in names]
    bp = ax.boxplot(boxdata, tick_labels=short_names, showfliers=False,
                    patch_artist=True, widths=0.6)
    colors = ["#E64B35", "#F39B7F", "#8491B4", "#4D5C84", "#4DBBD5", "#3B4992"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_ylabel("ARG1 expression")
    _, p = C.mwu(arg1[subsets["IL1R2+ neutrophils"]], arg1[subsets["IL1R2- neutrophils"]])
    p_str = "p < 1e-300" if p < 1e-300 else f"p = {p:.2e}"
    ax.set_title(f"ARG1 cell-source specificity\n(IL1R2+ vs IL1R2-, {p_str})")
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Cell subset")

    # (D)
    ax = fig.add_subplot(gs[1, 0])
    pos = gmdsc_df[gmdsc_df["IL1R2"] == "IL1R2+"]["GMDSC_score"]
    neg = gmdsc_df[gmdsc_df["IL1R2"] == "IL1R2-"]["GMDSC_score"]
    bp = ax.boxplot([neg, pos], tick_labels=["IL1R2-", "IL1R2+"], showfliers=False,
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#F39B7F"); bp["boxes"][1].set_facecolor("#E64B35")
    _, p = C.mwu(pos, neg)
    C.add_sig_annot(ax, 0, 1, max(pos.max(), neg.max()) * 1.06, 0.03, p)
    ax.set_ylabel("G-MDSC score")
    ax.set_title("G-MDSC signature")
    ax.set_xlabel("IL1R2 status")

    # (E)
    ax = fig.add_subplot(gs[1, 1])
    gs2 = list(co_genes)
    pcts = [coexp.loc[g, "pct_coexpressed_with_ARG1"] for g in gs2]
    ax.bar(range(len(gs2)), pcts, color="#E64B35", width=0.6)
    ax.set_xticks(range(len(gs2))); ax.set_xticklabels(gs2, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% co-expressed with ARG1\n(in IL1R2+ neutro)")
    ax.set_title("Immunosuppressive co-expression")
    ax.set_xlabel("Gene")

    # (F)
    ax = fig.add_subplot(gs[1, 2])
    idx_m = np.where(neut.values)[0][il1r2_pos]
    if len(idx_m) > 200:
        idx_m = idx_m[np.random.default_rng(0).choice(len(idx_m), 200, replace=False)]
    xx = il1r2[idx_m]; yy = arg1[idx_m]
    ax.scatter(xx, yy, s=6, c="#E64B35", alpha=0.4, rasterized=True)
    if len(xx) >= 3:
        rho, p, _ = C.spearman(xx, yy)
        ax.set_title(f"IL1R2 vs ARG1 (rho={rho:.2f})")
    ax.set_xlabel("IL1R2 expression"); ax.set_ylabel("ARG1 expression")

    C.label_panels(fig.axes, fontsize=15, dy=1.16)  # labels above the panels
    fig.tight_layout(rect=[0.02, 0, 1, 0.98])
    C.save_fig(fig, "fig2_neutrophil")

    # =====================================================================
    # Immunosuppressive co-expression module (IL1R2+ neutrophils)
    # =====================================================================
    module_genes = [g for g in ["ARG1", "PADI4", "CD47", "LGALS1", "TSPO", "ITGAM", "IL1R2"]
                    if g in list(adata.var_names)]
    expr = np.vstack([gene_col(adata, g)[neut.values][il1r2_pos] for g in module_genes])
    n_g = len(module_genes)
    corr_mat = np.zeros((n_g, n_g))
    for i in range(n_g):
        for j in range(n_g):
            rho, p, _ = C.spearman(expr[i], expr[j])
            corr_mat[i, j] = rho
    corr_mat_df = pd.DataFrame(corr_mat, index=module_genes, columns=module_genes)
    C.save_table(corr_mat_df.round(3), "fig2_coexpression_matrix")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sns.heatmap(corr_mat_df, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "Spearman rho"})
    ax.set_title("Immunosuppressive module (IL1R2+ neutrophils)")
    fig.tight_layout()
    C.save_fig(fig, "fig2_coexpression_network")

    print("Fig2 done.")


if __name__ == "__main__":
    main()