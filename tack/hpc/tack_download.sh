#!/bin/bash
# Downloads approved by the user on 2026-10-02 ("prova tutto quello che hai suggerito").
# sci-fate GSE131351 (300 MB) and sci-FATE2 GSE236512 (2.3 + 2.4 + 1.2 GB gz). Resumable, size-verified.
# Run with: nohup bash ~/tack/tack_download.sh > ~/tack/logs/download.log 2>&1 &
set -u
D=~/tack/data
mkdir -p $D/scifate $D/scifate2
get() {  # url dest
  local url=$1 dest=$2
  local remote=$(curl -sIL --max-time 60 "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  echo "$(date -Is) START $dest expected=$remote"
  wget -c -q --tries=20 --timeout=120 -O "$dest" "$url"
  local local_size=$(stat -c %s "$dest")
  if [ -n "$remote" ] && [ "$remote" = "$local_size" ]; then
    echo "$(date -Is) OK $dest $local_size"
  else
    echo "$(date -Is) SIZE_MISMATCH $dest local=$local_size remote=$remote"
  fi
}
G1=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131351/suppl
G2=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE236nnn/GSE236512/suppl
get $G1/filelist.txt $D/scifate/filelist.txt
get $G1/GSE131351_RAW.tar $D/scifate/GSE131351_RAW.tar
get $G2/GSE236512_processed_data_splicing.h5ad.gz $D/scifate2/GSE236512_processed_data_splicing.h5ad.gz
get $G2/GSE236512_processed_data_counting.h5ad.gz $D/scifate2/GSE236512_processed_data_counting.h5ad.gz
get $G2/GSE236512_processed_data_estimate.h5ad.gz $D/scifate2/GSE236512_processed_data_estimate.h5ad.gz
sha256sum $D/scifate/* $D/scifate2/* > ~/tack/logs/download_sha256.txt
echo "$(date -Is) DOWNLOAD_DONE"
