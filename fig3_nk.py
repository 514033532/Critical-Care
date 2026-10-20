"""
Fig 3 — NK cell subpopulation identification and function scores across stages.

Outputs:
  fig3_nk.pdf / .png
  tables/fig3_nk_scores_by_stage.csv
  tables/fig3_nk_subtype_composition.csv
  tables/fig3_nk_function_stage_stats.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__))
import common as C
import helpers as H


def main():
    C.setup_style()
    adata = H.load_adata()

    nk = H.nk_mask(adata)
    print(f"NK cells: {nk.sum()}")
    adata.obs["stage"] = [H.map_stage(d, g) for d, g in
                          zip(adata.obs["dataset"], adata.obs["group"])]
    nk_stage = adata.obs.loc[nk, "stage"].astype(str)

    # ---- function scores ----
    func_sets = {
        "Cytotoxicity": H.NK_CYTOTOX,
        "Exhaustion": H.NK_EXHAUST,
        "IFN_response": H.NK_IFN,
        "Activation": H.NK_ACTIVATION,
    }
    func_scores = {}
    for name, genes in func_sets.items():
        g = H.present_genes(adata, genes)
        func_scores[name] = H.module_score_z(adata, g)[nk]
        print(f"  {name}: {len(g)}/{len(genes)} genes")

    # ---- subtype assignment (argmax over subtype scores) ----
    subtype_sets = {
        "CD56dim_cytotoxic": H.NK_CD56DIM,
        "CD56bright_regulatory": H.NK_CD56BRIGHT,
        "Exhausted": H.NK_EXHAUSTED,
        "Proliferating": H.NK_PROLIF,
        "Tissue_homing": H.NK_HOMING,
    }
    S = np.vstack([H.module_score(adata, H.present_genes(adata, gs))[nk]
                   for gs in subtype_sets.values()])
    subtype_names = list(subtype_sets.keys())
    best = np.argmax(S, axis=0)
    subtypes = np.array([subtype_names[i] for i in best], dtype=object)

    # ---- export score table ----
    score_df = pd.DataFrame(func_scores)
    score_df["subtype"] = subtypes
    score_df["stage"] = nk_stage.values
    score_df["dataset"] = adata.obs.loc[nk, "dataset"].values
    score_df["sample"] = adata.obs.loc[nk, "sample"].values
    C.save_table(score_df, "fig3_nk_scores_per_cell")

    # ---- sample-level medians for stats ----
    smpl = score_df.groupby(["dataset", "sample", "stage"])[
        ["Cytotoxicity", "Exhaustion", "IFN_response", "Activation"]].median().reset_index()
    C.save_table(smpl, "fig3_nk_scores_sample_level")

    stages = [s for s in H.STAGE_ORDER if s in set(nk_stage)]
    stage_colors = {s: C.PALETTE.get(s, "#888888") for s in stages}
    stage_colors.update({"Healthy": C.PALETTE["Healthy"], "Acute_Sepsis": C.PALETTE["Sepsis"],
                         "Non-survivor": C.PALETTE["Non-survivor"],
                         "Survivor": C.PALETTE["Survivor"], "PICS": C.PALETTE["PICS"],
                         "ICU_non_sepsis": C.PALETTE["CI-NS"]})

    # ---- plot (a): function scores across stages ----
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0))
    for ax, (name, arr) in zip(axes.flat, func_scores.items()):
        data = [arr[nk_stage.values == s] for s in stages]
        bp = ax.boxplot(data, tick_labels=stages, showfliers=False, patch_artist=True, widths=0.6)
        for patch, s in zip(bp["boxes"], stages):
            patch.set_facecolor(stage_colors[s])
        ax.set_ylabel(f"{name} score")
        ax.set_xlabel("Stage")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        p = C.kruskal_pvals({s: arr[nk_stage.values == s] for s in stages})
        ax.set_title(f"{name}  (KW p={p:.2e})")
    C.label_panels(axes, fontsize=14)
    fig.tight_layout()
    C.save_fig(fig, "fig3_nk_function_scores")

    # ---- plot (b): subtype composition across stages ----
    comp = pd.crosstab(nk_stage.values, subtypes, normalize="index") * 100
    comp = comp.reindex(index=stages)
    C.save_table(comp.round(2), "fig3_nk_subtype_composition")
    sub_colors = {
        "CD56dim_cytotoxic": "#3B4992", "CD56bright_regulatory": "#4DBBD5",
        "Exhausted": "#E64B35", "Proliferating": "#00A087", "Tissue_homing": "#F28E2B",
    }
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    comp.plot(kind="bar", stacked=True, ax=ax,
              color=[sub_colors.get(c, "#BBBBBB") for c in comp.columns], width=0.7)
    ax.set_ylabel("NK subtype proportion (%)")
    ax.set_xlabel("Stage")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9, frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title("NK subpopulation composition")
    fig.tight_layout()
    C.save_fig(fig, "fig3_nk_subtype_composition")

    # ---- pairwise stats between key stages for exhaustion & cytotoxicity ----
    rows = []
    for score_name in ["Cytotoxicity", "Exhaustion", "IFN_response"]:
        arr = func_scores[score_name]
        for s1, s2 in combinations(stages, 2):
            _, p = C.mwu(arr[nk_stage.values == s1], arr[nk_stage.values == s2])
            rows.append({"score": score_name, "stage1": s1, "stage2": s2,
                         "n1": (nk_stage.values == s1).sum(), "n2": (nk_stage.values == s2).sum(),
                         "p_mwu": p})
    stat_df = pd.DataFrame(rows)
    stat_df["p_adj_fdr"] = C.fdr(stat_df["p_mwu"].values)
    C.save_table(stat_df, "fig3_nk_function_stage_stats")

    # =====================================================================
    # Subtype proportion vs function score (sample-level correlation)
    # =====================================================================
    sf = pd.DataFrame({"dataset": adata.obs.loc[nk, "dataset"].values,
                       "sample": adata.obs.loc[nk, "sample"].values,
                       "subtype": subtypes})
    subtype_prop = sf.pivot_table(index=["dataset", "sample"], columns="subtype",
                                  aggfunc="size", fill_value=0)
    subtype_prop = subtype_prop.div(subtype_prop.sum(axis=1), axis=0) * 100
    smpl_func = score_df.groupby(["dataset", "sample"])[
        ["Cytotoxicity", "Exhaustion", "IFN_response", "Activation"]].median()
    merged_sf = subtype_prop.join(smpl_func, how="inner")
    corr_rows = []
    for st in subtype_prop.columns:
        for fn in ["Cytotoxicity", "Exhaustion", "IFN_response", "Activation"]:
            rho, p, _ = C.spearman(merged_sf[st].values, merged_sf[fn].values)
            corr_rows.append({"subtype": st, "function": fn, "rho": rho, "p": p})
    corr_df = pd.DataFrame(corr_rows)
    C.save_table(corr_df, "fig3_subtype_function_correlation")
    corr_piv = corr_df.pivot(index="subtype", columns="function", values="rho")
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    sns.heatmap(corr_piv, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Spearman rho"})
    ax.set_title("NK subtype proportion vs function")
    fig.tight_layout()
    C.save_fig(fig, "fig3_nk_subtype_function")

    # =====================================================================
    # Exhaustion-marker stratification (double-phase exhaustion)
    # =====================================================================
    exh_markers = H.present_genes(adata, ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX",
                                          "TCF7", "CD244", "KLRC1", "CTLA4"])
    hm = np.zeros((len(exh_markers), len(stages)))
    for i, g in enumerate(exh_markers):
        gz = H.gene_col_z(adata, g)[nk]
        for j, s in enumerate(stages):
            hm[i, j] = np.mean(gz[nk_stage.values == s])
    hm_df = pd.DataFrame(hm, index=exh_markers, columns=stages)
    C.save_table(hm_df.round(3), "fig3_nk_exhaustion_markers")

    cyto_nk = func_scores["Cytotoxicity"]
    exh_nk = func_scores["Exhaustion"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    sns.heatmap(hm_df, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "z-score"})
    ax.set_xlabel("Stage")
    ax.set_ylabel("")
    ax.set_title("Exhaustion markers across stages")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax = axes[1]
    rng = np.random.default_rng(0)
    cell_idx = np.arange(len(nk_stage))
    if len(cell_idx) > 20000:
        cell_idx = rng.choice(cell_idx, 20000, replace=False)
    for s in stages:
        m = np.intersect1d(np.where(nk_stage.values == s)[0], cell_idx)
        ax.scatter(exh_nk[m], cyto_nk[m], s=3, c=stage_colors[s], alpha=0.5,
                   label=s, rasterized=True, linewidths=0)
    ax.set_xlabel("Exhaustion score (z)")
    ax.set_ylabel("Cytotoxicity score (z)")
    ax.set_title("Cytotoxicity vs exhaustion (NK cells)")
    ax.legend(fontsize=8, frameon=False, markerscale=3)
    C.label_panels(axes, fontsize=15)
    fig.tight_layout()
    C.save_fig(fig, "fig3_nk_exhaustion")

    print("Fig3 done.")


if __name__ == "__main__":
    main()
