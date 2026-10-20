"""
Fig 1 — Cross-dataset overview: UMAP (cell type / dataset) + composition.

Outputs:
  fig1_overview.pdf / .png
  tables/fig1_composition_dataset.csv
  tables/fig1_composition_stage.csv
  tables/fig1_umap_coords.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import helpers as H


def downsample(adata, n_per_dataset=8000, seed=0):
    rng = np.random.default_rng(seed)
    idx = []
    for ds in adata.obs["dataset"].unique():
        m = np.where(adata.obs["dataset"].values == ds)[0]
        k = min(n_per_dataset, len(m))
        idx.append(rng.choice(m, k, replace=False))
    return np.concatenate(idx)


def main():
    C.setup_style()
    adata = H.load_adata()
    print("annotated shape:", adata.shape)

    # ---- UMAP on a downsampled subset ----
    print("Computing UMAP on subset...", flush=True)
    sub_idx = downsample(adata, n_per_dataset=8000)
    sub = adata[sub_idx].copy()
    sc.pp.highly_variable_genes(sub, n_top_genes=2000, flavor="seurat_v3", subset=True)
    sc.pp.scale(sub, max_value=10)
    sc.tl.pca(sub, n_comps=30, svd_solver="arpack")
    sc.pp.neighbors(sub, n_neighbors=15, n_pcs=30)
    sc.tl.umap(sub)
    sub.obs["cell_type"] = sub.obs["cell_type"].astype(str)
    sub.obs["dataset"] = sub.obs["dataset"].astype(str)

    umap_df = pd.DataFrame({
        "UMAP1": sub.obsm["X_umap"][:, 0],
        "UMAP2": sub.obsm["X_umap"][:, 1],
        "cell_type": sub.obs["cell_type"].values,
        "dataset": sub.obs["dataset"].values,
        "group": sub.obs["group"].values,
    })
    C.save_table(umap_df, "fig1_umap_coords")

    ct_pal = {k: v for k, v in C.LINEAGE_PALETTE.items()}
    for ct in sub.obs["cell_type"].unique():
        ct_pal.setdefault(ct, "#BBBBBB")

    # ---- composition ----
    adata.obs["stage"] = [H.map_stage(d, g) for d, g in
                          zip(adata.obs["dataset"], adata.obs["group"])]
    comp_ds = adata.obs.groupby(["dataset", "cell_type"]).size().unstack(fill_value=0)
    comp_ds_prop = comp_ds.div(comp_ds.sum(axis=1), axis=0) * 100
    C.save_table(comp_ds, "fig1_composition_dataset_counts")
    C.save_table(comp_ds_prop.round(3), "fig1_composition_dataset_pct")
    comp_st = adata.obs.groupby(["stage", "cell_type"]).size().unstack(fill_value=0)
    comp_st_prop = comp_st.div(comp_st.sum(axis=1), axis=0) * 100
    C.save_table(comp_st, "fig1_composition_stage_counts")
    C.save_table(comp_st_prop.round(3), "fig1_composition_stage_pct")

    ct_tot = comp_ds.sum(axis=0).sort_values(ascending=False)
    order = [c for c in C.LINEAGE_ORDER if c in ct_tot.index]
    order += [c for c in ct_tot.index if c not in order]

    # ---- plot (2x2, panel-labelled) ----
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 11.0))

    # (A) UMAP by cell type
    ax = axes[0, 0]
    handles, labels_ct = [], []
    for ct in order:
        m = sub.obs["cell_type"].values == ct
        h = ax.scatter(sub.obsm["X_umap"][m, 0], sub.obsm["X_umap"][m, 1],
                       s=3.0, c=ct_pal[ct], label=ct, rasterized=True, linewidths=0)
        handles.append(h)
        labels_ct.append(ct)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("Cell type")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles, labels_ct, loc="upper right", frameon=True, facecolor="white",
              framealpha=0.85, fontsize=7, ncol=2, handletextpad=0.3,
              columnspacing=0.6, labelspacing=0.3, borderpad=0.4)

    # (B) UMAP by dataset
    ax = axes[0, 1]
    ds_order = sorted(set(sub.obs["dataset"]))
    ds_colors = plt.cm.tab10(np.linspace(0, 1, len(ds_order)))
    for i, ds in enumerate(ds_order):
        m = sub.obs["dataset"].values == ds
        ax.scatter(sub.obsm["X_umap"][m, 0], sub.obsm["X_umap"][m, 1],
                   s=3.0, c=[ds_colors[i]], label=ds, rasterized=True, linewidths=0)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("Dataset")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9,
              fontsize=7, handletextpad=0.3, borderpad=0.5)

    # (C) composition by dataset
    ax = axes[1, 0]
    props = comp_ds_prop.reindex(columns=order).T
    props.plot(kind="bar", stacked=True, ax=ax,
               color=[ct_pal.get(c, "#BBBBBB") for c in order], width=0.75, legend=False)
    ax.set_ylabel("Proportion (%)")
    ax.set_xlabel("Dataset")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_title("Composition by dataset")

    # (D) composition by stage
    ax = axes[1, 1]
    st_order = [s for s in H.STAGE_ORDER if s in comp_st.index]
    props2 = comp_st_prop.reindex(columns=order).T.reindex(columns=st_order)
    props2.plot(kind="bar", stacked=True, ax=ax,
                color=[ct_pal.get(c, "#BBBBBB") for c in order], width=0.75)
    ax.set_ylabel("Proportion (%)")
    ax.set_xlabel("Stage")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_title("Composition by stage")

    C.label_panels(axes, fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    C.save_fig(fig, "fig1_overview")

    print("Fig1 done.")


if __name__ == "__main__":
    main()