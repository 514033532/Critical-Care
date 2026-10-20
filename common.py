"""
Common utilities for the sepsis scRNA-seq analysis pipeline.

Shared SCI-compliant plotting style, statistical helpers, and output writers.
All figures are saved as both vector PDF (editable, embedded TrueType fonts)
and high-resolution PNG (300 dpi), with the underlying data exported to CSV.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = "/home/zhh2/Jiangjianrong/new-anlysis"
RAW_DIR = os.path.join(ROOT, "raw")
WORK_DIR = os.path.join(ROOT, "work")
FIG_DIR = os.path.join(ROOT, "results", "figures")
TAB_DIR = os.path.join(ROOT, "results", "tables")
LOG_DIR = os.path.join(ROOT, "logs")

ANNOTATED_H5AD = os.path.join(WORK_DIR, "adata_annotated.h5ad")

for _d in (WORK_DIR, FIG_DIR, TAB_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
# SCI figure style  (Arial / Helvetica, embedded TrueType, 300 dpi)
# ---------------------------------------------------------------------------
def setup_style():
    """Apply a journal-compliant matplotlib style (Arial, TrueType-embedded)."""
    # Arial is the de-facto standard; fall back to Liberation Sans (Helvetica metric)
    avail = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    fam = "Arial" if "Arial" in avail else "Liberation Sans"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [fam, "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,     # embed as TrueType (editable in Illustrator)
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return fam


def label_panels(axes, labels=None, fontsize=14, dx=-0.14, dy=1.04):
    """Add bold A/B/C/... panel labels to each axes (flattened)."""
    ax_list = np.asarray(axes, dtype=object).ravel().tolist()
    if labels is None:
        labels = [chr(65 + i) for i in range(len(ax_list))]
    for ax, lab in zip(ax_list, labels):
        ax.text(dx, dy, lab, transform=ax.transAxes, fontsize=fontsize,
                fontweight="bold", va="bottom", ha="left")


# Consistent, colorblind-friendly categorical palette (Okabe-Ito based)
PALETTE = {
    "Healthy": "#0072B2",
    "Sepsis": "#D55E00",
    "ARDS": "#CC79A7",
    "Survivor": "#009E73",
    "Non-survivor": "#E69F00",
    "PICS": "#8E44AD",
    "Acute": "#D55E00",
    "NHC": "#0072B2",
    "CI-NS": "#56B4E9",
    "CI-Sep": "#D55E00",
}
LINEAGE_PALETTE = {
    "Neutrophils": "#E64B35",
    "Immature_neutrophils": "#F39B7F",
    "NK cells": "#3B4992",
    "CD4 T cells": "#4DBBD5",
    "CD8 T cells": "#00A087",
    "T cells": "#7FB8A0",
    "B cells": "#E7B800",
    "Plasmablasts": "#C9A000",
    "Monocytes": "#A2A6B8",
    "Classical_monocytes": "#8491B4",
    "Non-classical_monocytes": "#4D5C84",
    "cDCs": "#9A8FCA",
    "pDCs": "#6A5ACD",
    "HSPCs": "#B09C85",
    "Platelets": "#B0B0B0",
    "Erythroid": "#C49C94",
    "Eosinophils": "#F28E2B",
    "Mast cells": "#8C564B",
    "Other": "#BBBBBB",
}

# Cell-type ordering for plots
LINEAGE_ORDER = [
    "Neutrophils", "Immature_neutrophils", "NK cells", "CD4 T cells", "CD8 T cells",
    "T cells", "B cells", "Plasmablasts", "Classical_monocytes",
    "Non-classical_monocytes", "Monocytes", "cDCs", "pDCs", "HSPCs",
    "Platelets", "Erythroid", "Eosinophils", "Mast cells", "Other",
]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
# Data source (GEO accession) for each figure — drawn as an on-figure footnote
# so every panel is traceable to the cohort(s) it uses.
_FIVE_COHORTS = "GSE151263/167363/216009/217906/290679"
FIGURE_SOURCE = {
    "fig1_overview": _FIVE_COHORTS,
    "fig2_neutrophil": "GSE216009",
    "fig2_coexpression_network": "GSE216009",
    "fig3_nk_function_scores": _FIVE_COHORTS,
    "fig3_nk_subtype_composition": _FIVE_COHORTS,
    "fig3_nk_subtype_function": _FIVE_COHORTS,
    "fig3_nk_exhaustion": _FIVE_COHORTS,
    "fig4_arg1_nk": "GSE216009",
    "fig5_cd38": _FIVE_COHORTS,
    "fig6_network": "GSE216009",
    "fig7_metabolism": "NK: " + _FIVE_COHORTS + " | CD4 T: GSE290679",
    "fig7_nk_cd4": _FIVE_COHORTS,
    "fig8_ml": "Train GSE290679→test GSE216009 | mortality GSE167363",
}

# Figures treated as supplementary — saved under results/figures/supplementary/.
SUPPLEMENTARY_FIGS = {
    "fig2_coexpression_network",
    "fig3_nk_subtype_composition",
    "fig3_nk_subtype_function",
    "fig3_nk_exhaustion",
    "fig7_nk_cd4",
}


def save_fig(fig, name):
    """Save a figure as PDF + PNG (300 dpi) into results/figures, with a
    data-source footnote (GEO accession) placed below the figure."""
    src = FIGURE_SOURCE.get(name)
    if src:
        # Negative figure coords put the footnote below the whole figure, so it can
        # never overlap axes, panel letters, or legends; bbox_inches="tight" includes it.
        fig.text(0.02, -0.02, f"Data: {src}",
                 ha="left", va="top", fontsize=7, style="italic", color="#333333",
                 bbox=dict(facecolor="white", edgecolor="#999999",
                           boxstyle="round,pad=0.25", alpha=0.9))
    out_dir = FIG_DIR if name not in SUPPLEMENTARY_FIGS else os.path.join(FIG_DIR, "supplementary")
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f"{name}.pdf")
    png = os.path.join(out_dir, f"{name}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [fig] {name}.pdf / .png")


def save_table(df, name, index=False):
    """Save a DataFrame to CSV into results/tables."""
    path = os.path.join(TAB_DIR, f"{name}.csv")
    df.to_csv(path, index=index)
    print(f"  [csv] {name}.csv  ({df.shape[0]} x {df.shape[1]})")
    return path


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def sig_label(p):
    """Return significance asterisks for a p-value."""
    if p is None or np.isnan(p):
        return "ns"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def mwu(a, b):
    """Mann-Whitney U test, returns (U, p)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return U, p


def kruskal_pvals(groups):
    """Kruskal-Wallis across groups (dict label -> array)."""
    arrs = [np.asarray(v, dtype=float) for v in groups.values()]
    arrs = [a[~np.isnan(a)] for a in arrs]
    arrs = [a for a in arrs if len(a) > 0]
    if len(arrs) < 2:
        return np.nan
    H, p = stats.kruskal(*arrs)
    return p


def spearman(x, y):
    """Spearman correlation with 95% CI (Fisher z)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan, np.nan, (np.nan, np.nan)
    rho, p = stats.spearmanr(x, y)
    # Fisher z transform for CI
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(len(x) - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return rho, p, (lo, hi)


def fdr(pvals):
    """Benjamini-Hochberg FDR."""
    pvals = np.asarray(pvals, dtype=float)
    mask = ~np.isnan(pvals)
    out = np.full_like(pvals, np.nan)
    if mask.sum() > 0:
        out[mask] = multipletests(pvals[mask], method="fdr_bh")[1]
    return out


def add_sig_annot(ax, x1, x2, y, h, p, text=None):
    """Draw a significance bracket between x1 and x2 at height y."""
    if p is None or np.isnan(p):
        return
    t = text if text is not None else sig_label(p)
    if t == "ns" and text is None:
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.7, c="black")
    ax.text((x1 + x2) / 2, y + h, t, ha="center", va="bottom", fontsize=9)
