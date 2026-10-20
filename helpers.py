"""
Shared analysis helpers: loading the annotated object, module scoring,
NK / neutrophil extraction, sample-level aggregation, and stage mapping.
"""
import os
import sys
import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, os.path.dirname(__file__))
import common as C


def load_adata():
    """Load the annotated AnnData (log-normalised float32)."""
    adata = ad.read_h5ad(C.ANNOTATED_H5AD)
    adata.var_names_make_unique()
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
    return adata


def gene_index(adata):
    return {g: i for i, g in enumerate(adata.var_names)}


def module_score(adata, genes, present_min=1):
    """Mean (native-normalised) expression of a gene set (0 for missing genes)."""
    gidx = [i for i, g in enumerate(adata.var_names) if g in genes]
    if len(gidx) < present_min:
        return np.zeros(adata.n_obs, dtype=np.float32)
    return np.asarray(adata.X[:, gidx].mean(axis=1)).ravel().astype(np.float32)


def module_score_z(adata, genes):
    """module_score() z-scored within each dataset (for cross-dataset comparability)."""
    s = module_score(adata, genes).astype(np.float64)
    out = np.zeros_like(s)
    for ds in np.unique(adata.obs["dataset"].values):
        m = adata.obs["dataset"].values == ds
        v = s[m]
        if v.std() > 1e-12:
            out[m] = (v - v.mean()) / v.std()
    return out


def gene_col(adata, gene):
    """Expression of one gene across all cells (0 if absent)."""
    names = list(adata.var_names)
    if gene not in names:
        return np.zeros(adata.n_obs, dtype=np.float32)
    col = adata.X[:, names.index(gene)]
    if hasattr(col, "toarray"):
        return np.asarray(col.toarray()).ravel().astype(np.float32)
    return np.asarray(col).ravel().astype(np.float32)


def gene_col_z(adata, gene):
    """gene_col() z-scored within each dataset."""
    s = gene_col(adata, gene).astype(np.float64)
    out = np.zeros_like(s)
    for ds in np.unique(adata.obs["dataset"].values):
        m = adata.obs["dataset"].values == ds
        v = s[m]
        if v.std() > 1e-12:
            out[m] = (v - v.mean()) / v.std()
    return out


def add_scores(adata, scores):
    """scores: dict name -> gene list. Adds columns '<name>_score' to obs."""
    for name, genes in scores.items():
        adata.obs[f"{name}_score"] = module_score(adata, genes)


def present_genes(adata, genes):
    v = set(adata.var_names)
    return [g for g in genes if g in v]


# ---------------------------------------------------------------------------
# Stage mapping (unified cross-dataset disease axis)
# ---------------------------------------------------------------------------
def map_stage(dataset, group):
    """Return a coarse disease-stage label for cross-dataset comparisons."""
    if group == "Healthy" or group == "NHC":
        return "Healthy"
    if group == "Non-survivor":
        return "Non-survivor"
    if group == "Survivor":
        return "Survivor"
    if group == "PICS":
        return "PICS"
    if group == "CI-NS":
        return "ICU_non_sepsis"
    # acute sepsis / ARDS
    if group in ("Sepsis", "ARDS", "CI-Sep"):
        return "Acute_Sepsis"
    return "Other"


STAGE_ORDER = ["Healthy", "Acute_Sepsis", "Survivor", "ICU_non_sepsis",
               "Non-survivor", "PICS"]


# ---------------------------------------------------------------------------
# Gene sets (filtered to present genes at call time)
# ---------------------------------------------------------------------------
NK_CYTOTOX = ["GZMA", "GZMB", "GZMH", "GZMK", "GNLY", "PRF1", "NKG7", "KLRD1", "FCGR3A"]
NK_EXHAUST = ["HAVCR2", "LAG3", "TIGIT", "PDCD1", "CD96", "KLRC1", "CD244", "CTLA4"]
NK_IFN = ["MX1", "ISG15", "IFIT1", "IFIT3", "STAT1", "IRF1", "OAS1", "IFI44L"]
NK_ACTIVATION = ["IFNG", "TNF", "KLRK1", "CD69", "GZMA", "PRF1"]

NK_CD56DIM = ["FCGR3A", "GZMB", "PRF1", "NKG7", "FGFBP2", "SPON2", "S1PR5"]
NK_CD56BRIGHT = ["NCAM1", "GZMK", "XCL1", "XCL2", "CCL5", "IL7R", "SELL"]
NK_EXHAUSTED = ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "CD96", "CTLA4", "TOX"]
NK_PROLIF = ["MKI67", "TOP2A", "STMN1", "TUBB", "UBE2C", "CENPF"]
NK_HOMING = ["CX3CR1", "ZEB2", "S1PR5", "ITGA4", "ITGB2"]

MTORC1 = ["MTOR", "RPTOR", "RHEB", "SLC7A5", "SLC3A2", "EIF4EBP1", "RPS6KB1"]
AMPK = ["PRKAA1", "PRKAA2", "PRKAB1", "PRKAB2", "STK11", "PRKAG1"]
NAD_METAB = ["NAMPT", "NMNAT1", "CD38", "NNMT", "NMNAT2", "NMNAT3"]

GMDSC = ["TSPO", "CD177", "ARG1", "IL1R2", "LGALS1", "CD33", "ITGAM"]
NEUTRO_IMMATURE = ["IL1R2", "OLFM4", "ELANE", "MPO", "AZU1", "DEFA3", "CEACAM8", "LCN2", "MME", "PADI4"]


# ---------------------------------------------------------------------------
# Cell extraction
# ---------------------------------------------------------------------------
def get_cells(adata, cell_types):
    """Boolean mask for coarse cell_type in a list."""
    return adata.obs["cell_type"].isin(cell_types).values


def nk_mask(adata):
    return (adata.obs["cell_type"] == "NK cells").values


def neutrophil_mask(adata):
    return (adata.obs["cell_type"] == "Neutrophils").values


def sample_level(adata, value_col, func="median"):
    """Aggregate a per-cell value to sample level; returns DataFrame [sample, value]."""
    df = adata.obs[["dataset", "sample", value_col]].copy()
    if func == "median":
        out = df.groupby(["dataset", "sample"])[value_col].median().reset_index()
    elif func == "mean":
        out = df.groupby(["dataset", "sample"])[value_col].mean().reset_index()
    out = out.rename(columns={value_col: "value"})
    return out
