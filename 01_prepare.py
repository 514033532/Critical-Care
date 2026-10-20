"""
01_prepare.py — Rebuild a correct, annotated merged AnnData from the six
individual source files (the shipped "inner merge" dropped GSE167363 /
GSE217906 expression because their gene names are GTF-format, and mixed raw
counts with z-scored matrices).

Per-dataset normalisation:
  * raw counts  (GSE167363, GSE290679) -> log1p(CPM/1e4)
  * z-scored    (GSE151263, GSE217906, GSE217906_New7) -> keep, sparsified |z|>=0.5
  * log-normal  (GSE216009) -> keep as-is

Cell types: GSE216009 uses its transferred `broad_annot`; all other datasets
are annotated by marker-gene scoring (argmax over lineage signatures).

Output: work/adata_annotated.h5ad (float32 csr)
"""
import os
import sys
import gzip
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import anndata as ad

sys.path.insert(0, os.path.dirname(__file__))
import common as C

# ---------------------------------------------------------------------------
# Sample -> group mappings
# ---------------------------------------------------------------------------
GSE151263_GROUP = {
    "ARDS1": "ARDS", "ARDS2": "ARDS", "ARDS3": "ARDS",
    "Sepsis1": "Sepsis", "Sepsis2": "Sepsis", "Sepsis3": "Sepsis", "Sepsis4": "Sepsis",
}
GSE167363_GROUP = {
    "HC1": "Healthy", "HC2": "Healthy",
    "P25": "Non-survivor", "NSES": "Non-survivor",
    "P50": "Survivor", "S2": "Survivor", "S3": "Survivor",
}
GSE217906_GROUP = {
    "PD-01": "PICS", "PD-02": "PICS", "PA-01": "PICS", "PA-02": "PICS",
    "CTFPA": "PICS", "YGAPA": "PICS",
    "AS-01": "Sepsis", "AS-02": "Sepsis",
    "LDQAS": "Sepsis", "LXHAS": "Sepsis", "MBBAS": "Sepsis",
    "HC-01": "Healthy", "HC-02": "Healthy", "HC-03": "Healthy",
    "WYXHC": "Healthy", "ZSQHC": "Healthy",
}

SOURCES = [
    ("GSE151263_all_samples_combined.h5ad", "GSE151263", "zscored"),
    ("GSE167363_all_samples_combined.h5ad", "GSE167363", "counts"),
    ("GSE216009_all_merged.h5ad", "GSE216009", "lognormal"),
    ("GSE217906_all_samples_combined.h5ad", "GSE217906", "zscored"),
    ("GSE217906_New7_all_samples_combined.h5ad", "GSE217906", "zscored"),
    ("GSE290679_all_merged_final.h5ad", "GSE290679", "counts"),
]

MARKERS = {
    "NK cells": ["NKG7", "GNLY", "KLRD1", "KLRF1", "GZMB", "PRF1", "FGFBP2", "SPON2", "FCGR3A"],
    "CD4 T cells": ["CD3D", "CD3E", "CD4", "IL7R", "LTB", "TRAC", "CD40LG"],
    "CD8 T cells": ["CD3D", "CD3E", "CD8A", "CD8B", "GZMK", "CCL5", "GZMA"],
    "B cells": ["MS4A1", "CD79A", "CD79B", "BANK1", "CD19"],
    "Plasmablasts": ["MZB1", "JCHAIN", "XBP1", "IGHG1", "IGHM", "DERL3"],
    "Classical_monocytes": ["CD14", "LYZ", "S100A8", "S100A9", "VCAN", "FCN1", "S100A12"],
    "Non-classical_monocytes": ["FCGR3A", "MS4A7", "LST1", "IFITM3", "CSF1R", "LILRB2"],
    "cDCs": ["CD1C", "FCER1A", "HLA-DQA1", "HLA-DPB1", "HLA-DRA"],
    "pDCs": ["LILRA4", "IRF7", "PLD4", "IL3RA", "TCF4", "CLEC4C"],
    "Platelets": ["PPBP", "PF4", "GP9", "TUBB1", "ITGA2B"],
    "Erythroid": ["HBA1", "HBA2", "HBB", "AHSP", "ALAS2", "HBM"],
    "HSPCs": ["SPINK2", "PRSS57", "AVP", "GATA2"],
    "Neutrophils": ["FCGR3B", "CSF3R", "S100A8", "S100A9", "CXCR2", "G0S2"],
}
BROAD2COARSE = {
    "Mature_neutrophils": "Neutrophils", "Degranulating_neutrophils": "Neutrophils",
    "Apoptosing_neutrophils": "Neutrophils", "Immature_neutrophils": "Neutrophils",
    "Neutrophil_progenitors": "Neutrophils", "NK cells": "NK cells",
    "CD4_T_cells": "CD4 T cells", "CD8_T_cells": "CD8 T cells", "Cycling_TNK": "T cells",
    "B_cells": "B cells", "Plasmablasts": "Plasmablasts",
    "Classical_monocytes": "Monocytes", "Non-classical_monocytes": "Monocytes",
    "cDCs": "DCs", "pDCs": "DCs", "HSPCs": "HSPCs", "Platelets": "Platelets",
    "Mast_cells/eosiniophils": "Mast cells", "Eosinophils": "Eosinophils",
}


def parse_gse290679_groups():
    gsm_list, grp_list = None, None
    with gzip.open(os.path.join(C.WORK_DIR, "GSE290679_matrix.txt.gz"), "rt") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if line.startswith("!Sample_geo_accession"):
                gsm_list = [x.strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1") and '\t"group: ' in line:
                grp_list = [x.strip('"').replace("group: ", "").strip() for x in line.split("\t")[1:]]
    return {g: gr for g, gr in zip(gsm_list, grp_list)} if gsm_list and grp_list else {}


def clean_gene(name):
    return name.split("\t")[1].strip() if "\t" in name else name.strip()


def read_col(f, key):
    g = f["obs"][key]
    if isinstance(g, h5py.Group):
        cats = [c.decode() for c in g["categories"][()]]
        return [cats[c] for c in g["codes"][:]]
    v = g[()]
    return [x.decode() if isinstance(x, bytes) else x for x in v]


def read_genes(fname):
    with h5py.File(os.path.join(C.RAW_DIR, fname), "r") as f:
        gr = f["var"]["_index"][:]
        return [clean_gene(g.decode() if isinstance(g, bytes) else g) for g in gr]


def load_aligned(fname, ds, norm, universe):
    """Load one source file, subset to `universe`, normalise -> (csr X, barcodes, sample)."""
    path = os.path.join(C.RAW_DIR, fname)
    with h5py.File(path, "r") as f:
        genes = [clean_gene(g.decode() if isinstance(g, bytes) else g)
                 for g in f["var"]["_index"][:]]
        bc = read_col(f, "_index")
        Xgrp = f["X"]
        is_sparse = isinstance(Xgrp, h5py.Group)
        if is_sparse:
            data = Xgrp["data"][:]
            indices = Xgrp["indices"][:]
            indptr = Xgrp["indptr"][:]
            shape = tuple(Xgrp.attrs["shape"])
            X = sp.csr_matrix((data, indices, indptr), shape=shape).astype(np.float32)
        else:
            X = np.asarray(Xgrp[()]).astype(np.float32)  # dense numpy

    # subset to universe columns (before csr conversion when dense), remap to universe width
    g2i = {g: i for i, g in enumerate(genes)}
    col_idx = np.array([g2i.get(g, -1) for g in universe], dtype=np.int64)
    keep = col_idx >= 0
    keep_cols = col_idx[keep].astype(np.int32)
    uni_pos = np.where(keep)[0].astype(np.int32)
    n = X.shape[0]
    if is_sparse:
        sub = X[:, keep_cols].tocsr()
        full = sp.csr_matrix((sub.data, uni_pos[sub.indices], sub.indptr),
                             shape=(n, len(universe)))
    else:
        sub = X[:, keep_cols]  # numpy column slice
        tmp = sp.csr_matrix(sub)
        full = sp.csr_matrix((tmp.data, uni_pos[tmp.indices], tmp.indptr),
                             shape=(n, len(universe)))
    del X

    # normalise
    if norm == "counts":
        rs = np.asarray(full.sum(axis=1)).ravel()
        rs[rs == 0] = 1
        full = full.multiply((1e4 / rs)[:, None]).tocsr()
        full.data = np.log1p(full.data)
    elif norm == "zscored":
        full.data[np.abs(full.data) < 0.5] = 0
        full.eliminate_zeros()

    # sample extraction
    if ds == "GSE290679":
        with h5py.File(path, "r") as f:
            sample = [s.split("_")[0] if s != "NA" else "NA" for s in read_col(f, "sample")]
    elif ds == "GSE216009":
        with h5py.File(path, "r") as f:
            sample = read_col(f, "sample_id")
    else:
        sample = [b.split("_")[1] if "_" in b else b for b in bc]

    return full, bc, sample


def main():
    C.setup_style()
    print("Pass 1: reading gene names...", flush=True)
    gene_blocks = []
    for fname, ds, norm in SOURCES:
        genes = read_genes(fname)
        gene_blocks.append(genes)
        print(f"  {fname}: {len(genes)} genes")
    all_genes = [g for gs in gene_blocks for g in gs]
    from collections import Counter
    cnt = Counter(all_genes)
    universe = sorted([g for g, c in cnt.items() if c >= 3 and g and g != "NA"])
    print(f"Gene universe (present in >=3 sources): {len(universe)}")

    print("Pass 2: loading + aligning...", flush=True)
    gse290679_map = parse_gse290679_groups()
    pieces = []
    obs_list = []
    for fname, ds, norm in SOURCES:
        X, bc, sample = load_aligned(fname, ds, norm, universe)
        n = X.shape[0]
        groups = np.empty(n, dtype=object)
        for j in range(n):
            s = sample[j]
            if ds == "GSE290679":
                groups[j] = gse290679_map.get(s, "Unknown")
            elif ds == "GSE151263":
                groups[j] = GSE151263_GROUP.get(s, "Unknown")
            elif ds == "GSE167363":
                groups[j] = GSE167363_GROUP.get(s, "Unknown")
            elif ds == "GSE217906":
                groups[j] = GSE217906_GROUP.get(s, "Unknown")
            else:
                groups[j] = "Sepsis"  # GSE216009 filled after transfer
        obs_list.append(pd.DataFrame({"dataset": ds, "sample": sample, "group": groups},
                                     index=bc))
        pieces.append(X)
        print(f"  {fname}: {n} cells, nnz={X.nnz}")

    print("Concatenating...", flush=True)
    Xall = sp.vstack(pieces).tocsr()
    obs_all = pd.concat(obs_list, axis=0)
    obs_all.index.name = "cell"
    del pieces

    # ---- transfer GSE216009 annotations ----
    print("Transferring GSE216009 annotations...", flush=True)
    with h5py.File(os.path.join(C.RAW_DIR, "GSE216009_all_merged.h5ad"), "r") as f:
        bc216 = read_col(f, "_index")
        broad = read_col(f, "broad_annot")
        fine = read_col(f, "fine_annot")
        diag = read_col(f, "diagnosis")
        sid = read_col(f, "sample_id")
    m216 = pd.DataFrame({"broad_annot": broad, "fine_annot": fine,
                         "diagnosis": diag, "sample_id": sid}, index=bc216)
    m216 = m216[~m216.index.duplicated(keep="first")]
    g216 = obs_all["dataset"] == "GSE216009"
    obs_all.loc[g216, "broad_annot"] = m216.reindex(obs_all.index[g216])["broad_annot"].values
    obs_all.loc[g216, "fine_annot"] = m216.reindex(obs_all.index[g216])["fine_annot"].values
    obs_all.loc[g216, "diagnosis"] = m216.reindex(obs_all.index[g216])["diagnosis"].values
    obs_all.loc[g216, "sample"] = m216.reindex(obs_all.index[g216])["sample_id"].values
    obs_all.loc[g216, "group"] = np.where(
        obs_all.loc[g216, "diagnosis"].astype(str).isin(["HV"]), "Healthy", "Sepsis")

    # ---- cell type annotation ----
    print("Annotating cell types...", flush=True)
    ct = np.empty(Xall.shape[0], dtype=object)
    ct_fine = np.empty(Xall.shape[0], dtype=object)
    uni_idx = {g: i for i, g in enumerate(universe)}
    marker_idx = {k: [uni_idx[g] for g in gs if g in uni_idx] for k, gs in MARKERS.items()}

    ct[g216.values] = obs_all.loc[g216, "broad_annot"].map(BROAD2COARSE).fillna("Other").values
    ct_fine[g216.values] = obs_all.loc[g216, "broad_annot"].fillna("Other").values

    non216 = ~g216.values
    X_non = Xall[non216]
    score_names = list(marker_idx.keys())
    S = []
    for k in score_names:
        idxs = marker_idx[k]
        if not idxs:
            S.append(np.zeros(X_non.shape[0], dtype=np.float32))
        else:
            S.append(np.asarray(X_non[:, idxs].mean(axis=1)).ravel().astype(np.float32))
    S = np.vstack(S)
    best = np.argmax(S, axis=0)
    assign = np.array([score_names[i] for i in best], dtype=object)
    assign[S[best, np.arange(S.shape[1])] <= 0] = "Other"
    tlab = {"CD4 T cells", "CD8 T cells"}
    if "CD3D" in uni_idx and "CD3E" in uni_idx:
        cd3 = np.asarray(X_non[:, [uni_idx["CD3D"], uni_idx["CD3E"]]].mean(axis=1)).ravel()
        assign[np.isin(assign, list(tlab)) & (cd3 <= 0)] = "Other"
    ct[non216] = assign
    ct_fine[non216] = assign
    # unify coarse vocabulary (monocyte/DC subclasses -> coarse) to match GSE216009
    COARSE_MAP = {"Classical_monocytes": "Monocytes", "Non-classical_monocytes": "Monocytes",
                  "cDCs": "DCs", "pDCs": "DCs"}
    ct[non216] = np.array([COARSE_MAP.get(a, a) for a in ct[non216]], dtype=object)

    obs_all["cell_type"] = ct
    obs_all["cell_type_fine"] = ct_fine

    comp = obs_all.groupby(["dataset", "cell_type"]).size().unstack(fill_value=0)
    print(comp.to_string())
    C.save_table(obs_all.groupby(["dataset", "group", "sample"]).size().rename("n_cells").reset_index(),
                 "meta_sample_counts")

    print("Saving annotated h5ad...", flush=True)
    adata = ad.AnnData(X=Xall, obs=obs_all, var=pd.DataFrame(index=universe))
    adata.var_names_make_unique()
    adata.write_h5ad(C.ANNOTATED_H5AD, compression="gzip")
    print("Saved:", C.ANNOTATED_H5AD, adata.shape)


if __name__ == "__main__":
    main()
