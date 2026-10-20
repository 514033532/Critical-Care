"""
Cell-cell communication analysis with LIANA (CellChat / CellPhoneDB / Connectome).

LIANA is a Python-native framework that wraps multiple ligand-receptor methods
(including CellChat's rank-based approach) and reads AnnData directly.

Usage:
    python3 scripts/run_cellchat.py

Outputs (into results/tables/):
    cellchat_consensus.csv      — CellChat (rank-based) results
    cellphonedb_consensus.csv   — CellPhoneDB (magnitude-based) results
"""
import os
import sys
import numpy as np
import anndata as ad
import liana
import liana.method as mt

sys.path.insert(0, os.path.dirname(__file__))
import common as C

# tune these for speed vs. rigor
N_PER_CT = 3000      # max cells sampled per cell type (permutation test is O(n_cells * n_perms))
N_PERMS = 500        # permutation iterations (1000 is the conventional default)
GROUPBY = "cell_type"


def downsample(adata, n_per_ct=N_PER_CT, seed=0):
    rng = np.random.default_rng(seed)
    idx = []
    for ct in adata.obs[GROUPBY].unique():
        m = np.where(adata.obs[GROUPBY].values == ct)[0]
        k = min(n_per_ct, len(m))
        idx.append(rng.choice(m, k, replace=False))
    return adata[np.concatenate(idx)].copy()


def main():
    C.setup_style()
    adata = ad.read_h5ad(C.ANNOTATED_H5AD)
    adata.var_names_make_unique()
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()

    print("annotated shape:", adata.shape)
    sub = downsample(adata)
    print("downsampled:", sub.shape, "| cell types:", sub.obs[GROUPBY].nunique())

    # ---- CellChat (rank-based) ----
    # NOTE: results are saved under NEW names (cellchat_*) so the existing
    # custom fig6 ligand-receptor analysis is NOT overwritten — both are kept for comparison.
    print("Running CellChat (rank-based)...", flush=True)
    res_cc = mt.cellchat(
        sub, groupby=GROUPBY, resource_name="consensus",
        expr_prop=0.1, use_raw=False, n_perms=N_PERMS, seed=42,
        inplace=False, verbose=True,
    )
    C.save_table(res_cc, "cellchat_consensus")
    print("CellChat done:", res_cc.shape)

    # dot plot — restricted to the myeloid -> lymphoid axis (manuscript focus) so the
    # panel stays readable instead of plotting all 15x15 cell-type pairs.
    myeloid = ["Neutrophils", "Monocytes"]
    lymphoid = ["NK cells", "CD4 T cells", "CD8 T cells", "T cells"]
    res_cc_focus = res_cc[
        res_cc["source"].isin(myeloid) & res_cc["target"].isin(lymphoid)
    ].copy()
    print("CellChat focus subset:", res_cc_focus.shape,
          "| cell-type pairs:", res_cc_focus.groupby(["source", "target"]).ngroups)

    # dot plot (for comparison against fig6_network)
    fig = liana.pl.dotplot(
        liana_res=res_cc_focus,
        colour="lr_probs",         # CellChat communication probability
        size="lr_probs",
        top_n=15,
        orderby="lr_probs",        # rank interactions by communication probability
        orderby_ascending=False,   # strongest first
        figure_size=(9, 7),
        return_fig=True,
    )
    # liana.pl.dotplot returns a plotnine `ggplot` object, so save directly
    _supp = os.path.join(C.FIG_DIR, "supplementary")
    os.makedirs(_supp, exist_ok=True)
    fig.save(os.path.join(_supp, "cellchat_dotplot.png"), width=9, height=7, dpi=300)
    fig.save(os.path.join(_supp, "cellchat_dotplot.pdf"), width=9, height=7)
    print("  [fig] cellchat_dotplot.png / .pdf")

    # ---- CellPhoneDB (magnitude-based) ----
    print("Running CellPhoneDB (magnitude-based)...", flush=True)
    res_cp = mt.cellphonedb(
        sub, groupby=GROUPBY, resource_name="consensus",
        expr_prop=0.1, use_raw=False, n_perms=N_PERMS, seed=42,
        inplace=False, verbose=True,
    )
    C.save_table(res_cp, "cellphonedb_consensus")
    print("CellPhoneDB done:", res_cp.shape)

    print("All done.")


if __name__ == "__main__":
    main()
