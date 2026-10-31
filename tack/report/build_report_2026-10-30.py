"""Assemble deliverables/ot_couplings_deliverable_2026-10-30 (+ .zip): the report (md / docx / pdf), its figures, the
pre-registrations, the result JSONs and our own scripts and tests. No raw data and no Dileo & Sottoriva code.
Styling and the Word step reuse cfm_grn_v2/scripts_dev/docx_style.py and the pandoc + Word COM block of
build_deliverable.py.
Usage (any directory): python tack/report/build_report_2026-10-30.py
"""
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
tack = here.parent
proj = tack.parent
name = "ot_couplings_deliverable_2026-10-30"
dst = proj / "deliverables" / name
REPORT = "REPORT_OT_couplings_vs_snapshots"
FOOTER = "Where optimal transport helps on real single-cell time courses  ·  G. Bambini  ·  30 October 2026"

PREREGS = ["2026-10-22-dileo-task2-distbridge-prereg.md", "2026-10-25-atlas-confirmation-prereg.md",
           "2026-10-26-embryo-age-prereg.md", "2026-10-27-zebrafish-replication-prereg.md",
           "2026-10-29-dynmix-prereg.md", "2026-10-29-larry-prereg.md", "2026-10-30-larry-invivo-prereg.md",
           "2026-10-30-larry-cytokine-prereg.md"]
RESULTS = ["dileo_task2/distbridge/pancreas-human/summary.json", "dileo_task2/distbridge/gastrulation-human/summary.json",
           "atlas/confirm.json", "atlas/age/age_confirm.json", "atlas/dynmix_atlas.json", "zf/confirm_zf.json",
           "zf/dynmix_zf.json", "larry/larry_confirm.json", "larry/larry_robust.json", "larry/larry_sharp.json",
           "larry_vivo/larry_vivo_confirm.json", "larry_vivo/larry_vivo_sanity.json", "larry_vivo/meta.json",
           "larry_cyto/larry_cyto_confirm.json", "larry_cyto/meta.json", "larry_cyto/design_729828.log",
           "larry_cyto/prep_729843.log", "larry_cyto/confirm_729844.log"]
CODE = ["sinkhorn.py", "dist_bridge.py", "task2_forecasters.py", "dileo_harness.py", "dileo_task2.py", "dileo_rescore.py",
        "interp_bench_frozen.py", "atlas_prep.py", "atlas_confirm.py", "ery_prep.py", "ery_forecast.py",
        "ery_forecast2.py", "diag_gap.py", "diag_gap_zf.py", "age_confirm.py", "zf_meta.py", "zf_prep.py", "dynmix.py",
        "larry_prep.py", "larry_confirm.py", "larry_robust.py", "larry_sharp.py", "larry_vivo_design.py",
        "larry_vivo_prep.py", "larry_vivo_confirm.py", "larry_vivo_sanity.py", "larry_vivo_typecheck.py",
        "larry_vivo_order.py", "larry_cyto_design.py", "larry_cyto_prep.py", "larry_cyto_confirm.py"]


def clear_tree(path: Path, attempts: int = 6) -> None:
    """Remove a previous build; the Drive client can hold files open, so retry before failing."""
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            for f in path.rglob("*"):
                try:
                    os.chmod(f, stat.S_IWRITE)
                except OSError:
                    pass
            time.sleep(1.5)


def copy(src: Path, dest: Path) -> None:
    if not src.exists():
        print("MISSING", src)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)


dst.mkdir(parents=True, exist_ok=True)
for f in [p for p in dst.rglob("*") if p.is_file()]:      # delete files; the Drive client may lock the folders
    os.chmod(f, stat.S_IWRITE)
    f.unlink()
for d in sorted((p for p in dst.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
    try:
        d.rmdir()
    except OSError:
        pass
shutil.copy(here / f"{REPORT}.md", dst / f"{REPORT}.md")
subprocess.run([sys.executable, str(here / "make_figures_2026-10-30.py"), str(dst / "figures")], check=True)
for p in PREREGS:
    copy(tack / "prereg" / p, dst / "prereg" / p)
for p in RESULTS:
    copy(tack / "results" / p, dst / "results" / p)
for p in CODE:
    copy(tack / "hpc" / p, dst / "code" / p)
for f in sorted((tack / "tests").glob("test_*.py")):
    copy(f, dst / "code" / "tests" / f.name)
for f in (here / "make_figures_2026-10-30.py", here / Path(__file__).name):
    copy(f, dst / "code" / "report" / f.name)
copy(proj / "docs" / "superpowers" / "decisions.md", dst / "docs" / "decisions_log_it.md")

# --- docx / pdf via pandoc (styled reference) + Word, as in cfm_grn_v2/scripts_dev/build_deliverable.py ----------
sys.path.insert(0, str(proj / "cfm_grn_v2" / "scripts_dev"))
from docx_style import make_reference_docx, polish_docx  # noqa: E402

ref = make_reference_docx(dst / "_reference.docx")
docx = dst / f"{REPORT}.docx"
subprocess.run(["pandoc", f"{REPORT}.md", "-o", docx.name, "--from", "markdown", "--to", "docx", "--reference-doc", ref.name,
                "--toc", "--toc-depth", "2", "-M", "lang=en-GB"], check=True, cwd=str(dst))
ref.unlink()
polish_docx(docx, FOOTER)
pdf = dst / f"{REPORT}.pdf"
ps = f'''$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0
$d = $w.Documents.Open("{docx}"); $d.Fields.Update() | Out-Null
foreach ($t in $d.TablesOfContents) {{ $t.Update() | Out-Null }}
$d.Save(); $d.SaveAs2("{pdf}", 17); $d.Close(); $w.Quit()'''
r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
print("pdf:", "ok" if pdf.exists() else f"failed: {r.stderr[:300]}")

zip_path = shutil.make_archive(str(proj / "deliverables" / name), "zip", root_dir=dst.parent, base_dir=dst.name)
print("deliverable folder:", dst)
print("zip:", zip_path, round(Path(zip_path).stat().st_size / 1e6, 1), "MB")
