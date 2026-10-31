# Convert the Saunders 2023 zebrafish reference counts (GSE202639_reference_raw_counts.RDS.gz, a sparse genes x cells
# matrix) into raw binary CSC vectors readable by numpy: zf_i.int32, zf_p.float64, zf_x.float32, zf_dims.txt,
# zf_genes.txt, zf_cells.txt in ~/rafm_data/saunders2023/csc/.
suppressMessages(library(Matrix))
src <- path.expand("~/rafm_data/saunders2023/GSE202639_reference_raw_counts.RDS.gz")
out <- path.expand("~/rafm_data/saunders2023/csc")
dir.create(out, showWarnings = FALSE)
plain <- file.path(out, "raw_counts.RDS")
if (!file.exists(plain)) system(paste("gunzip -c", shQuote(src), ">", shQuote(plain)))
m <- tryCatch(readRDS(plain), error = function(e) { cat("plain readRDS failed:", conditionMessage(e), "\n");
  system(paste("head -c 8", shQuote(plain), "| od -c")); readRDS(gzcon(file(plain, "rb"))) })
cat("class", class(m), "dims", dim(m), "\n")
if (!inherits(m, "dgCMatrix")) m <- as(m, "CsparseMatrix")
writeBin(as.integer(m@i), file.path(out, "zf_i.int32"), size = 4)
writeBin(as.numeric(m@p), file.path(out, "zf_p.float64"), size = 8)
writeBin(as.numeric(m@x), file.path(out, "zf_x.float32"), size = 4)
writeLines(as.character(dim(m)), file.path(out, "zf_dims.txt"))
writeLines(rownames(m), file.path(out, "zf_genes.txt"))
writeLines(colnames(m), file.path(out, "zf_cells.txt"))
cat("nnz", length(m@x), "done\n")
