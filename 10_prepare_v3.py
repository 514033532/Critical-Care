"""
10_prepare_v3.py — Process GSE279452 (pediatric + adult sepsis) and GSE175453
(late sepsis / CCI) raw 10x data into annotated AnnData for independent
validation (v3).  Reuses the marker-gene scoring vocabulary from 01_prepare.py.

Output:
  work/v3_GSE279452.h5ad   (log1p CPM, cell_type annotated)
  work/v3_GSE175453.h5ad
"""
import os
import sys
import gzip
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import anndata as ad

sys.path.insert(0, os.path.dirname(__file__))
import common as C


# ---------------------------------------------------------------------------
# Group mapping
# ---------------------------------------------------------------------------
def map_279452(sample):
    s = sample.upper()
    if s.startswith("HC"):
        return "Adult_Healthy"
    if s.startswith("IC"):
        return "Adult_ICU"
    if s.startswith("PHC"):
        return "Pediatric_Healthy"
    if s.startswith("PIC"):
        return "Pediatric_ICU"
    if "-PS" in s:
        return "Pediatric_Sepsis"
    return "Adult_Sepsis"


def map_175453(sample):
    s = sample.upper()
    if s.startswith("H"):  # HC1, HC2, H20, H27, H30
        return "Healthy"
    if s == "SEP1":
        return "Sepsis_RR"
    return "Sepsis_CCI"


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
COARSE = {"Classical_monocytes": "Monocytes", "Non-classical_monocytes": "Monocytes",
          "cDCs": "DCs", "pDCs": "DCs"}


def find_triplets(rawdir, sep):
    mats = glob.glob(os.path.join(rawdir, f"*{sep}matrix.mtx.gz"))
    out = []
    for mp in mats:
        base = mp[: -len(sep + "matrix.mtx.gz")]
        fp = base + sep + "features.tsv.gz"
        bp = base + sep + "barcodes.tsv.gz"
        sample = os.path.basename(base).split("_", 1)[1]
        out.append((sample, mp, fp, bp))
    return sorted(out)


def read_genes(fp):
    with gzip.open(fp, "rt") as f:
        return [ln.rstrip("\n").split("\t")[1] for ln in f]


def read_matrix(mp):
    return sio.mmread(mp).T.tocsr().astype(np.float32)  # -> cells x genes


def annotate(X, genes):
    g2i = {g: i for i, g in enumerate(genes)}
    S = []
    for k in list(MARKERS):
        idx = [g2i[g] for g in MARKERS[k] if g in g2i]
        if idx:
            S.append(np.asarray(X[:, idx].mean(axis=1)).ravel().astype(np.float32))
        else:
            S.append(np.zeros(X.shape[0], dtype=np.float32))
    S = np.vstack(S)
    best = np.argmax(S, axis=0)
    assign = np.array([list(MARKERS)[i] for i in best], dtype=object)
    assign[S[best, np.arange(S.shape[1])] <= 0] = "Other"
    if "CD3D" in g2i and "CD3E" in g2i:
        cd3 = np.asarray(X[:, [g2i["CD3D"], g2i["CD3E"]]].mean(axis=1)).ravel()
        assign[np.isin(assign, ["CD4 T cells", "CD8 T cells"]) & (cd3 <= 0)] = "Other"
    return np.array([COARSE.get(a, a) for a in assign], dtype=object)


def process(name, rawdir, sep, map_fn):
    triplets = find_triplets(rawdir, sep)
    print(f"{name}: {len(triplets)} samples", flush=True)
    # pass 1: gene universe (intersection across samples)
    universe = set(read_genes(triplets[0][2]))
    for _, _, fp, _ in triplets[1:]:
        universe &= set(read_genes(fp))
    universe = sorted(universe)
    print(f"  gene universe: {len(universe)}", flush=True)

    Xs, obs_list = [], []
    for sample, mp, fp, bp in triplets:
        X = read_matrix(mp)
        genes = read_genes(fp)
        g2i = {g: i for i, g in enumerate(genes)}
        keep_idx = np.array([g2i.get(g, -1) for g in universe], dtype=np.int64)
        X = X[:, keep_idx].tocsr()
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
        keep_cell = n_genes >= 200
        X = X[keep_cell]
        with gzip.open(bp, "rt") as f:
            bc = [ln.rstrip("\n") for ln in f]
        bc = [b for b, k in zip(bc, keep_cell) if k]
        rs = np.asarray(X.sum(axis=1)).ravel(); rs[rs == 0] = 1
        X = X.multiply((1e4 / rs)[:, None]).tocsr(); X.data = np.log1p(X.data)
        grp = map_fn(sample)
        Xs.append(X)
        obs_list.append(pd.DataFrame({"sample": sample, "group": grp,
                                      "barcode": [f"{sample}_{b}" for b in bc]}))
        print(f"  {sample} -> {grp}: {X.shape[0]} cells", flush=True)
    Xall = sp.vstack(Xs).tocsr()
    obs = pd.concat(obs_list, axis=0)
    obs["cell_type"] = annotate(Xall, universe)
    obs["dataset"] = name
    adata = ad.AnnData(X=Xall, obs=obs, var=pd.DataFrame(index=universe))
    adata.var_names_make_unique()
    out = os.path.join(C.WORK_DIR, f"v3_{name}.h5ad")
    adata.write_h5ad(out, compression="gzip")
    print(f"saved {out}  {adata.shape}", flush=True)
    comp = obs.groupby(["group", "cell_type"]).size().unstack(fill_value=0)
    print(comp.to_string(), flush=True)


def main():
    C.setup_style()
    process("GSE279452", os.path.join(C.ROOT, "GSE279452_RAW"), ".", map_279452)
    process("GSE175453", os.path.join(C.ROOT, "GSE175453_RAW"), "_", map_175453)
    print("ALL DONE")


if __name__ == "__main__":
    main()
