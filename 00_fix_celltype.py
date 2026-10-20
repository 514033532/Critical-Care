"""One-off fix: unify coarse cell_type vocabulary (monocyte/DC subclasses -> coarse)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import common as C
import anndata as ad

COARSE = {"Classical_monocytes": "Monocytes", "Non-classical_monocytes": "Monocytes",
          "cDCs": "DCs", "pDCs": "DCs"}

adata = ad.read_h5ad(C.ANNOTATED_H5AD)
before = adata.obs["cell_type"].value_counts()
adata.obs["cell_type"] = adata.obs["cell_type"].replace(COARSE)
after = adata.obs["cell_type"].value_counts()
print("=== before ==="); print(before.to_string())
print("=== after ==="); print(after.to_string())
adata.write_h5ad(C.ANNOTATED_H5AD, compression="gzip")
print("saved")
