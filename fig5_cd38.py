"""
Fig 5 — CD38 biphasic model: stage-dependent CD38–NK function relationship.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    nk = H.nk_mask(adata)
    adata.obs["stage"] = [H.map_stage(d, g) for d, g in
                          zip(adata.obs["dataset"], adata.obs["group"])]

    cd38 = H.gene_col_z(adata, "CD38")
    pdcd1 = gene_col(adata, "PDCD1")
    cyto = H.module_score_z(adata, H.present_genes(adata, H.NK_CYTOTOX))
    nad = H.module_score(adata, H.present_genes(adata, H.NAD_METAB))

    stage_nk = adata.obs.loc[nk, "stage"].astype(str)
    cd38_nk = cd38[nk]
    pdcd1_nk = pdcd1[nk]
    cyto_nk = cyto[nk]
    nad_nk = nad[nk]

    stages = [s for s in H.STAGE_ORDER if s in set(stage_nk)]
    stage_colors = {"Healthy": C.PALETTE["Healthy"], "Acute_Sepsis": C.PALETTE["Sepsis"],
                    "Survivor": C.PALETTE["Survivor"], "Non-survivor": C.PALETTE["Non-survivor"],
                    "PICS": C.PALETTE["PICS"], "ICU_non_sepsis": C.PALETTE["CI-NS"]}

    corr_rows = []
    for s in stages:
        m = stage_nk.values == s
        x = cd38_nk[m]; y = cyto_nk[m]
        rho, p, ci = C.spearman(x, y)
        corr_rows.append({
            "stage": s, "n_NK": int(m.sum()),
            "CD38_median": float(np.median(x)),
            "CD38_pct_expressed": float((x > 0).mean() * 100),
            "rho_CD38_vs_cytotoxicity": rho, "p": p,
            "ci_low": ci[0], "ci_high": ci[1],
        })
    cdf = pd.DataFrame(corr_rows)
    C.save_table(cdf, "fig5_cd38_stage_correlation")
    print(cdf.to_string())

    coexp_rows = []
    for s in stages:
        m = stage_nk.values == s
        x = cd38_nk[m]; p = pdcd1_nk[m]
        coexp_rows.append({
            "stage": s,
            "CD38pos_PD1pos_pct": float((((x > 0) & (p > 0)).mean()) * 100),
            "CD38pos_pct": float((x > 0).mean() * 100),
            "PD1pos_pct": float((p > 0).mean() * 100),
        })
    coexp_df = pd.DataFrame(coexp_rows)
    C.save_table(coexp_df, "fig5_cd38_pd1_coexpression")
    print(coexp_df.to_string())

    fig = plt.figure(figsize=(10.0, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.75, wspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    data = [cd38_nk[stage_nk.values == s] for s in stages]
    bp = ax.boxplot(data, tick_labels=stages, showfliers=False, patch_artist=True, widths=0.6)
    for patch, s in zip(bp["boxes"], stages):
        patch.set_facecolor(stage_colors[s])
    ax.set_ylabel("CD38 expression (z-score)")
    ax.set_xlabel("Stage")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    p = C.kruskal_pvals({s: cd38_nk[stage_nk.values == s] for s in stages})
    ax.set_title(f"NK CD38 (KW p={p:.1e})")

    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(0)
    for s in stages:
        m = np.where(stage_nk.values == s)[0]
        if len(m) > 800:
            m = m[rng.choice(len(m), 800, replace=False)]
        ax.scatter(cd38_nk[m], cyto_nk[m], s=3, c=stage_colors[s], alpha=0.5,
                   label=s, rasterized=True, linewidths=0)
        if len(m) >= 3:
            z = np.polyfit(cd38_nk[m], cyto_nk[m], 1)
            xs = np.array([cd38_nk[m].min(), cd38_nk[m].max()])
            ax.plot(xs, np.polyval(z, xs), color=stage_colors[s], lw=1.4)
    ax.set_xlabel("CD38 expression (z-score)"); ax.set_ylabel("NK cytotoxicity (z-score)")
    ax.legend(fontsize=8, frameon=False, markerscale=3, loc="upper right")
    ax.set_title("CD38 vs cytotoxicity by stage")

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(range(len(stages)), cdf["rho_CD38_vs_cytotoxicity"].values,
           color=[stage_colors[s] for s in stages], width=0.6)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("rho (CD38 vs cytotoxicity)")
    ax.set_xlabel("Stage")
    ax.set_title("Correlation direction across stages")
    for i, r in enumerate(cdf.itertuples()):
        ax.text(i, r.rho_CD38_vs_cytotoxicity + (0.02 if r.rho_CD38_vs_cytotoxicity >= 0 else -0.07),
                f"p={r.p:.2g}", ha="center", fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(range(len(stages)), coexp_df["CD38pos_PD1pos_pct"].values,
           color=[stage_colors[s] for s in stages], width=0.6)
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("CD38+PD-1+ NK cells (%)")
    ax.set_xlabel("Stage")
    ax.set_title("CD38/PD-1 double-positive NK")

    C.label_panels(fig.axes, fontsize=15)
    fig.tight_layout(rect=[0.02, 0, 1, 0.98])
    C.save_fig(fig, "fig5_cd38")

    print("Fig5 done.")


if __name__ == "__main__":
    main()