"""
Fig 6 — Myeloid–NK–T immunosuppression network (ligand–receptor, expression-based).

Curated ligand-receptor communication scores between myeloid sources and
lymphoid targets, computed in GSE216009 (whole blood), compared Healthy vs Sepsis.

Outputs:
  fig6_network.pdf / .png
  tables/fig6_ligand_receptor_scores.csv
  tables/fig6_communication_by_group.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import common as C
import helpers as H


# curated ligand -> receptor(s)  (immunosuppression-focused)
LR_PAIRS = [
    ("ARG1", ["SLC7A1", "SLC7A2", "SLC7A5"]),          # arginine depletion
    ("CD274", ["PDCD1"]),                               # PD-L1 -> PD-1
    ("MIF", ["CD74", "CD44"]),                          # MIF -> CD74
    ("TGFB1", ["TGFBR1", "TGFBR2"]),                    # TGFb
    ("HLA-E", ["KLRD1", "KLRC2", "KLRC3"]),             # HLA-E -> CD94/NKG2
    ("LGALS9", ["HAVCR2"]),                             # Gal-9 -> TIM-3
    ("CD47", ["SIRPA"]),                                # CD47 -> SIRPa
    ("LGALS1", ["CD69", "PTPRC"]),                      # Gal-1 -> CD45/CD69
    ("CXCL8", ["CXCR1", "CXCR2"]),                      # IL-8
    ("IL10", ["IL10RA", "IL10RB"]),                     # IL-10
]


def main():
    C.setup_style()
    adata = H.load_adata()
    g216 = adata.obs["dataset"] == "GSE216009"

    # ---- pre-extract expression of all needed genes (one pass) ----
    needed = ["IL1R2"]
    for lig, recs in LR_PAIRS:
        needed.append(lig)
        needed += recs
    needed = list(dict.fromkeys(needed))
    names = list(adata.var_names)
    present = [g for g in needed if g in names]
    gidx = [names.index(g) for g in present]
    E = np.asarray(adata.X[:, gidx].toarray(), dtype=np.float32)  # (n_cells, k)
    g2c = {g: i for i, g in enumerate(present)}

    def me(gene, mask):
        if gene not in g2c:
            return 0.0
        return max(0.0, float(E[mask, g2c[gene]].mean()))

    def comm(lig, recs, src, tgt):
        l = me(lig, src)
        r = 0.0
        for rr in recs:
            if rr in g2c:
                r = max(r, me(rr, tgt))
        if l == 0.0 and r == 0.0:
            return 0.0
        return l * r

    il1r2_col = g2c["IL1R2"] if "IL1R2" in g2c else None
    il1r2 = E[:, il1r2_col] if il1r2_col is not None else np.zeros(E.shape[0])

    neut = (g216 & (adata.obs["cell_type"] == "Neutrophils")).values
    il1r2_pos_neut = neut & (il1r2 > 0)
    il1r2_neg_neut = neut & (il1r2 == 0)
    mono = (g216 & (adata.obs["cell_type"] == "Monocytes")).values
    nk = (g216 & (adata.obs["cell_type"] == "NK cells")).values
    cd4 = (g216 & (adata.obs["cell_type"] == "CD4 T cells")).values
    cd8 = (g216 & (adata.obs["cell_type"] == "CD8 T cells")).values

    sources = {"IL1R2+ neutro": il1r2_pos_neut, "IL1R2- neutro": il1r2_neg_neut,
               "Monocytes": mono, "CD4 T": cd4}
    targets = {"NK": nk, "CD4 T": cd4, "CD8 T": cd8}

    # ---- communication scores (all cells) ----
    rows = []
    for sname, sm in sources.items():
        for tname, tm in targets.items():
            for lig, recs in LR_PAIRS:
                rows.append({"source": sname, "target": tname, "ligand": lig,
                             "receptor": "/".join(recs),
                             "comm_score": comm(lig, recs, sm, tm)})
    sdf = pd.DataFrame(rows)
    C.save_table(sdf, "fig6_ligand_receptor_scores")

    # ---- Healthy vs Sepsis comparison for key edges ----
    diag = adata.obs.loc[g216, "diagnosis"].astype(str).values
    is_hv = np.repeat(False, adata.n_obs); is_hv[g216.values] = (diag == "HV")
    is_sep = np.repeat(False, adata.n_obs); is_sep[g216.values] = (diag != "HV")

    key_edges = [
        ("IL1R2+ neutro", "NK", "ARG1"), ("IL1R2+ neutro", "CD4 T", "ARG1"),
        ("IL1R2+ neutro", "NK", "CD274"), ("IL1R2+ neutro", "CD4 T", "CD274"),
        ("IL1R2+ neutro", "NK", "MIF"), ("IL1R2+ neutro", "CD4 T", "MIF"),
        ("Monocytes", "NK", "TGFB1"), ("Monocytes", "CD4 T", "TGFB1"),
    ]
    comp_rows = []
    for sname, tname, lig in key_edges:
        recs = [p[1] for p in LR_PAIRS if p[0] == lig][0]
        sc_hv = comm(lig, recs, sources[sname] & is_hv, targets[tname] & is_hv)
        sc_sep = comm(lig, recs, sources[sname] & is_sep, targets[tname] & is_sep)
        comp_rows.append({"source": sname, "target": tname, "ligand": lig,
                          "edge": f"{sname} → {tname} ({lig})", "Healthy": sc_hv,
                          "Sepsis": sc_sep,
                          "log2FC": np.log2((sc_sep + 1e-9) / (sc_hv + 1e-9))})
    cdf = pd.DataFrame(comp_rows)
    C.save_table(cdf, "fig6_communication_by_group")
    print(cdf.to_string())

    # ---- plot ----
    fig = plt.figure(figsize=(12.5, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.6)

    src_short = {"IL1R2+ neutro": "IL1R2+ Neu", "IL1R2- neutro": "IL1R2- Neu",
                 "Monocytes": "Mono", "CD4 T": "CD4 T"}
    tgt_short = {"NK": "NK", "CD4 T": "CD4", "CD8 T": "CD8"}
    sdf["edge_short"] = [src_short[s] + "→" + tgt_short[t]
                         for s, t in zip(sdf["source"], sdf["target"])]

    ax_heat = fig.add_subplot(gs[0, 0])
    piv = sdf.groupby(["ligand", "edge_short"])["comm_score"].max().unstack()
    edges = sorted(sdf["edge_short"].unique())
    piv = piv.reindex(columns=edges)
    im = ax_heat.imshow(np.log1p(piv.values), aspect="auto", cmap="viridis")
    ax_heat.set_xticks(range(len(edges))); ax_heat.set_xticklabels(edges, rotation=45, ha="right", fontsize=9)
    ax_heat.set_yticks(range(len(piv.index))); ax_heat.set_yticklabels(piv.index, fontsize=10)
    ax_heat.set_xlabel("Source → Target")
    ax_heat.set_title("Ligand-receptor communication (GSE216009)")
    plt.colorbar(im, ax=ax_heat, shrink=0.7, label="log1p score")

    ax_bar = fig.add_subplot(gs[0, 1])
    cdf["edge_short"] = [src_short[s] + "→" + tgt_short[t] + f" ({lig})"
                         for s, t, lig in zip(cdf["source"], cdf["target"], cdf["ligand"])]
    order = cdf.sort_values("log2FC")["edge_short"].values
    vals = cdf.set_index("edge_short").loc[order, "log2FC"].values
    colors = ["#0072B2" if v <= 0 else "#D55E00" for v in vals]
    ax_bar.barh(range(len(order)), vals, color=colors)
    ax_bar.set_yticks(range(len(order))); ax_bar.set_yticklabels(order, fontsize=9)
    ax_bar.axvline(0, color="black", lw=0.9)
    ax_bar.set_xlabel("log2FC (Sepsis vs Healthy)")
    ax_bar.set_title("Immunosuppressive edge changes")

    # label only the two data panels (skip the colorbar axes)
    C.label_panels([ax_heat, ax_bar], fontsize=14)
    fig.tight_layout()
    C.save_fig(fig, "fig6_network")
    print("Fig6 done.")


if __name__ == "__main__":
    main()
