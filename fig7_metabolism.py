"""
Fig 7 — mTORC1/AMPK metabolic mechanism + immune-clock stage mapping.

Outputs:
  fig7_metabolism.pdf / .png
  tables/fig7_nk_metabolism_stage.csv
  tables/fig7_cd4_metabolism.csv
  tables/fig7_immune_clock.csv
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
    adata.obs["stage"] = [H.map_stage(d, g) for d, g in
                          zip(adata.obs["dataset"], adata.obs["group"])]

    mtor = H.module_score(adata, H.present_genes(adata, H.MTORC1))
    ampk = H.module_score(adata, H.present_genes(adata, H.AMPK))
    ratio_raw = (mtor + 1e-6) / (ampk + 1e-6)
    nrf1 = H.gene_col_z(adata, "NRF1")
    cpt1a = H.gene_col_z(adata, "CPT1A")
    # z-score the ratio within dataset for cross-dataset comparability
    ratio = np.zeros_like(ratio_raw, dtype=np.float64)
    for ds in np.unique(adata.obs["dataset"].values):
        m = adata.obs["dataset"].values == ds
        v = ratio_raw[m]
        if v.std() > 1e-12:
            ratio[m] = (v - v.mean()) / v.std()

    nk = H.nk_mask(adata)
    cd4 = (adata.obs["cell_type"] == "CD4 T cells").values

    stage_colors = {"Healthy": C.PALETTE["Healthy"], "Acute_Sepsis": C.PALETTE["Sepsis"],
                    "Survivor": C.PALETTE["Survivor"], "Non-survivor": C.PALETTE["Non-survivor"],
                    "PICS": C.PALETTE["PICS"], "ICU_non_sepsis": C.PALETTE["CI-NS"]}
    stages = [s for s in H.STAGE_ORDER if (adata.obs.loc[nk, "stage"] == s).any()]

    # ---- NK metabolism by stage ----
    nk_stage = adata.obs.loc[nk, "stage"].astype(str)
    ratio_nk = ratio[nk]; nrf1_nk = nrf1[nk]; cpt1a_nk = cpt1a[nk]
    rows = []
    for s in stages:
        m = nk_stage.values == s
        rows.append({
            "stage": s, "n_NK": int(m.sum()),
            "mTORC1_AMPK_ratio_median": float(np.median(ratio_nk[m])),
            "NRF1_median": float(np.median(nrf1_nk[m])),
            "CPT1A_median": float(np.median(cpt1a_nk[m])),
        })
    nk_met = pd.DataFrame(rows)
    C.save_table(nk_met, "fig7_nk_metabolism_stage")
    print(nk_met.to_string())

    # ---- CD4 T metabolism in GSE290679 ----
    g290 = adata.obs["dataset"] == "GSE290679"
    cd4_290 = cd4 & g290
    cd4_rows = []
    for grp in ["NHC", "CI-NS", "CI-Sep"]:
        m = cd4_290 & (adata.obs["group"] == grp).values
        cd4_rows.append({
            "group": grp, "n_CD4": int(m.sum()),
            "mTORC1_AMPK_ratio_median": float(np.median(ratio[m])),
            "mTORC1_median": float(np.median(mtor[m])),
            "AMPK_median": float(np.median(ampk[m])),
            "NRF1_median": float(np.median(nrf1[m])),
            "CPT1A_median": float(np.median(cpt1a[m])),
        })
    cd4_met = pd.DataFrame(cd4_rows)
    C.save_table(cd4_met, "fig7_cd4_metabolism")
    print(cd4_met.to_string())

    # ---- immune clock: exhaustion/cytotoxicity trajectory ----
    cyto = H.module_score(adata, H.present_genes(adata, H.NK_CYTOTOX))
    exhaust = H.module_score(adata, H.present_genes(adata, H.NK_EXHAUST))
    clock_rows = []
    for s in stages:
        m = nk_stage.values == s
        clock_rows.append({
            "stage": s,
            "cytotoxicity_median": float(np.median(cyto[nk][m])),
            "exhaustion_median": float(np.median(exhaust[nk][m])),
        })
    clock = pd.DataFrame(clock_rows)
    C.save_table(clock, "fig7_immune_clock")
    print(clock.to_string())

    # ---- plot ----
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.5))

    # (a) NK mTORC1/AMPK ratio across stages
    ax = axes[0, 0]
    data = [ratio_nk[nk_stage.values == s] for s in stages]
    bp = ax.boxplot(data, tick_labels=stages, showfliers=False, patch_artist=True, widths=0.6)
    for patch, s in zip(bp["boxes"], stages):
        patch.set_facecolor(stage_colors[s])
    ax.set_ylabel("NK mTORC1/AMPK ratio (z-score)")
    ax.set_xlabel("Stage")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    p = C.kruskal_pvals({s: ratio_nk[nk_stage.values == s] for s in stages})
    ax.set_title(f"NK metabolic balance (KW p={p:.2e})")

    # (b) NRF1 / CPT1A across stages (grouped)
    ax = axes[0, 1]
    x = np.arange(len(stages)); w = 0.35
    nrf = [np.median(nrf1_nk[nk_stage.values == s]) for s in stages]
    cpt = [np.median(cpt1a_nk[nk_stage.values == s]) for s in stages]
    ax.bar(x - w/2, nrf, w, label="NRF1", color="#3B4992")
    ax.bar(x + w/2, cpt, w, label="CPT1A", color="#E64B35")
    ax.set_xticks(x); ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Median expression (NK)")
    ax.set_title("Mitochondrial regulators")
    ax.legend(fontsize=9, frameon=False, loc="upper right", ncol=2)
    ax.set_ylim(-0.38, 0.10)

    # (c) CD4 T metabolism in GSE290679
    ax = axes[1, 0]
    ax.bar(range(3), cd4_met["mTORC1_AMPK_ratio_median"].values,
           color=[C.PALETTE["NHC"], C.PALETTE["CI-NS"], C.PALETTE["CI-Sep"]], width=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Healthy", "ICU non-sep", "ICU sepsis"], fontsize=9)
    ax.set_xlabel("Group")
    ax.set_ylabel("CD4 T mTORC1/AMPK ratio (z-score)")
    ax.set_title("CD4 T metabolism (GSE290679)")
    ax.set_ylim(-0.65, 0.02)

    # (d) immune clock trajectory
    ax = axes[1, 1]
    x = np.arange(len(stages))
    ax.plot(x, clock["cytotoxicity_median"].values, marker="o", color="#3B4992",
            label="Cytotoxicity", lw=1.5)
    ax.plot(x, clock["exhaustion_median"].values, marker="s", color="#E64B35",
            label="Exhaustion", lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Median NK score")
    ax.set_title("Immune-clock stage trajectory")
    ax.legend(fontsize=9, frameon=False)
    ax.set_ylim(0, 2.8)

    C.label_panels(axes, fontsize=15, dy=1.16)  # labels above the panels
    fig.tight_layout(h_pad=2.5, w_pad=1.8)
    C.save_fig(fig, "fig7_metabolism")

    # =====================================================================
    # NK function vs CD4 T metabolism (sample-level correlation)
    # =====================================================================
    nk_s = adata.obs.loc[nk, ["dataset", "sample"]].copy()
    nk_s["cyto"] = cyto[nk]
    nk_s["exhaust"] = exhaust[nk]
    nk_med = nk_s.groupby(["dataset", "sample"])[["cyto", "exhaust"]].median()
    cd4_s = adata.obs.loc[cd4, ["dataset", "sample"]].copy()
    cd4_s["ratio"] = ratio[cd4]
    cd4_med = cd4_s.groupby(["dataset", "sample"])["ratio"].median()
    samp_stage = adata.obs.groupby(["dataset", "sample"])["stage"].first()
    merged_nc = nk_med.join(cd4_med, how="inner").join(samp_stage, how="left")
    C.save_table(merged_nc.reset_index(), "fig7_nk_cd4_sample_level")

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for s in sorted(set(merged_nc["stage"].dropna())):
        m = merged_nc["stage"].values == s
        ax.scatter(merged_nc.loc[m, "exhaust"], merged_nc.loc[m, "ratio"],
                   s=30, c=stage_colors.get(s, "#888888"), label=s,
                   edgecolors="black", linewidths=0.4, zorder=3)
    xv = merged_nc["exhaust"].values; yv = merged_nc["ratio"].values
    rho, p, _ = C.spearman(xv, yv)
    m = ~(np.isnan(xv) | np.isnan(yv))
    if m.sum() >= 3:
        z = np.polyfit(xv[m], yv[m], 1)
        xs = np.linspace(np.nanmin(xv), np.nanmax(xv), 50)
        ax.plot(xs, np.polyval(z, xs), color="grey", lw=1.0, ls="--")
    ax.set_xlabel("NK exhaustion score (median)")
    ax.set_ylabel("CD4 T mTORC1/AMPK ratio (z, median)")
    ax.set_title(f"NK vs CD4 T (rho={rho:.2f}, p={p:.3f})")
    ax.legend(title="Stage", fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    C.save_fig(fig, "fig7_nk_cd4")

    print("Fig7 done.")


if __name__ == "__main__":
    main()
