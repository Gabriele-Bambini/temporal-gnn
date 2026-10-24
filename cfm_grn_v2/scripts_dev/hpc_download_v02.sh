#!/bin/bash
# Downloads approved by the user on 2026-09-25 (plan docs/superpowers/plans/2026-09-25-rafm-next-step-proposal.md).
# Run on the HPC login node with nohup; every file is checked against the byte size recorded before approval.
set -u
BASE=~/rafm_data
LOG=$BASE/download_v02.log
mkdir -p $BASE/eb_published $BASE/veres2019 $BASE/pijuan_sala2019
echo "start $(date -Is)" >> $LOG

get () {  # url dest expected_bytes
  url=$1; dest=$2; want=$3
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$want" ]; then
    echo "SKIP $dest (already $want B)" >> $LOG; return 0
  fi
  curl -L -f -sS --retry 5 --retry-delay 10 -o "$dest.part" "$url" >> $LOG 2>&1
  got=$(stat -c %s "$dest.part" 2>/dev/null || echo 0)
  if [ "$got" = "$want" ]; then
    mv "$dest.part" "$dest"
    echo "OK   $dest $got B sha256 $(sha256sum "$dest" | cut -d' ' -f1)" >> $LOG
  else
    echo "FAIL $dest got $got B, expected $want B" >> $LOG
  fi
}

get https://github.com/KrishnaswamyLab/TrajectoryNet/raw/master/data/eb_velocity_v5.npz \
    $BASE/eb_published/eb_velocity_v5.npz 28142608
get https://ftp.ncbi.nlm.nih.gov/geo/series/GSE114nnn/GSE114412/suppl/GSE114412_Stage_5.all.cell_metadata.tsv.gz \
    $BASE/veres2019/GSE114412_Stage_5.all.cell_metadata.tsv.gz 1764292
get https://ftp.ncbi.nlm.nih.gov/geo/series/GSE114nnn/GSE114412/suppl/GSE114412_Stage_5.all.processed_counts.tsv.gz \
    $BASE/veres2019/GSE114412_Stage_5.all.processed_counts.tsv.gz 108394226
get https://content.cruk.cam.ac.uk/jmlab/atlas_data/meta.tab.gz $BASE/pijuan_sala2019/meta.tab.gz 4941002
get https://content.cruk.cam.ac.uk/jmlab/atlas_data/genes.tsv.gz $BASE/pijuan_sala2019/genes.tsv.gz 228898
get https://content.cruk.cam.ac.uk/jmlab/atlas_data/raw_counts.mtx.gz $BASE/pijuan_sala2019/raw_counts.mtx.gz 1527653430
echo "done $(date -Is)" >> $LOG
