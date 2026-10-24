"""Dump code cells (and short text outputs) of the Chronocell notebooks so the exact erythroid settings can be read."""
import json, sys, pathlib

for path in sys.argv[1:]:
    p = pathlib.Path(path)
    if not p.exists():
        print(f"### MISSING {p}")
        continue
    nb = json.loads(p.read_text(encoding="utf-8"))
    print(f"\n######## {p.name} ########")
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code":
            print(f"\n--- [code {i}] ---\n{src}")
            for out in cell.get("outputs", []):
                txt = out.get("text") or out.get("data", {}).get("text/plain")
                if txt:
                    t = "".join(txt)
                    print(f"--- [out {i}] ---\n{t[:1500]}")
        elif cell.get("cell_type") == "markdown" and src.strip():
            print(f"\n--- [md {i}] ---\n{src[:600]}")
