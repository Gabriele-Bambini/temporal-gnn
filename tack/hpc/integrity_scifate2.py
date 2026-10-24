"""Pre-freeze integrity check (metadata only): obs_names / var_names identical across the sci-FATE2 files."""
import anndata
import numpy as np

D = "/home/3393519/tack/data/scifate2/"
files = ["GSE236512_processed_data_counting.h5ad", "GSE236512_processed_data_splicing.h5ad",
         "GSE236512_processed_data_estimate.h5ad"]
ads = [anndata.read_h5ad(D + f, backed="r") for f in files]
ref = ads[0]
for f, ad in zip(files[1:], ads[1:]):
    print(f, "obs identical:", bool(np.array_equal(ref.obs_names, ad.obs_names)),
          "| var identical:", bool(np.array_equal(ref.var_names, ad.var_names)),
          "| obs meta identical:", bool(ref.obs[["timepoint", "rep"]].equals(ad.obs[["timepoint", "rep"]])))
print("n mt- genes:", int(np.sum(ref.var_names.str.startswith("mt-"))))
print("obs_names unique:", ref.obs_names.is_unique, "| var_names unique:", ref.var_names.is_unique)
for ad in ads:
    ad.file.close()
