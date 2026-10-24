"""C2: in low-signal ('quiet') transitions, does the delta-method noise R explain the l1 change variance and its
negative lag-1 covariance, by CP10k class? And after removing the library/cell-size term m^2 CV2_lib from s^2?

y_i = L_{i+1} - L_i (all genes, float64). Signal share_i = (var_g y_i - mean_g(R_i + R_{i+1})) / var_g y_i.
Selection: share < 0.2 at both i and i-1 (computed with the uncorrected R, as the question defines; the corrected-R
selection is also reported). Class = CP10k class of bin i. Ratios over (genes in class) x (selected transitions):
  var ratio = sum (y_i - c)^2 / sum (R_i + R_{i+1});  lag ratio = -sum (y_{i-1} - c')(y_i - c) / sum R_i,
centring c, c' = mean of y_i, y_{i-1} over the same (class, transition) gene set ('cls'), or over all genes of the
transition ('all'). Pure noise with no signal autocorrelation gives 1 for both.
"""
import json
import os

import numpy as np

import common as C


def ratios(e, Rm, sel, centre="cls"):
    Y = e.Yd
    out = {}
    for c, lab in enumerate(C.CP_LABELS):
        num_v = den_v = num_l = den_l = 0.0
        ng = []
        for i in sel:
            g = C.cp_class(e.CP[:, i]) == c
            if g.sum() < 3:
                continue
            y, y0 = Y[i], Y[i - 1]
            if centre == "cls":
                yc, y0c = y[g] - y[g].mean(), y0[g] - y0[g].mean()
            else:
                yc, y0c = (y - y.mean())[g], (y0 - y0.mean())[g]
            num_v += (yc ** 2).sum()
            den_v += (Rm[g, i] + Rm[g, i + 1]).sum()
            num_l += -(y0c * yc).sum()
            den_l += Rm[g, i].sum()
            ng.append(int(g.sum()))
        out[lab] = {"var_ratio": round(num_v / den_v, 3) if den_v > 0 else None,
                    "lag_ratio": round(num_l / den_l, 3) if den_l > 0 else None,
                    "genes_per_transition": round(float(np.mean(ng)), 1) if ng else 0,
                    "ss_y": float(num_v)}
    tot = sum(v["ss_y"] for v in out.values())
    for v in out.values():
        v["ss_share"] = round(v.pop("ss_y") / tot, 4) if tot > 0 else None
    return out


def share(e, Rm):
    Y = e.Yd
    v = Y.var(1)
    nm = np.array([(Rm[:, i] + Rm[:, i + 1]).mean() for i in range(e.T)])
    return (v - nm) / v


res = {}
for name in C.NAMES:
    e = C.Ext(name)
    sh, shc = share(e, e.R), share(e, e.Rc)
    sel = [i for i in range(1, e.T) if sh[i] < 0.2 and sh[i - 1] < 0.2]
    selc = [i for i in range(1, e.T) if shc[i] < 0.2 and shc[i - 1] < 0.2]
    allt = list(range(1, e.T))
    # variant: CV2_lib floored at 0 (the 5th percentile is negative in small bins, where subtracting ADDS variance)
    Rc0 = e.r_of(np.maximum(e.s2 - e.m ** 2 * np.maximum(e.cv2lib, 0)[None, :], 0.0))
    big = [i for i in sel if min(e.n[i - 1], e.n[i], e.n[i + 1]) >= 100]
    r = {"cv2lib_per_bin": np.round(e.cv2lib, 4).tolist(), "n_cells": np.round(e.n).tolist(),
         "selected_bins>=100": big,
         "signal_share_R": np.round(sh, 3).tolist(), "signal_share_Rcorr": np.round(shc, 3).tolist(),
         "selected_R (share<0.2 at i and i-1)": sel, "selected_Rcorr": selc,
         "Rcorr_over_R_by_class(sel)": {}}
    for c, lab in enumerate(C.CP_LABELS):
        num = den = 0.0
        for i in (sel or allt):
            g = C.cp_class(e.CP[:, i]) == c
            num += e.Rc[g, i].sum()
            den += e.R[g, i].sum()
        r["Rcorr_over_R_by_class(sel)"][lab] = round(num / den, 3) if den > 0 else None
    for tag, tr in (("selected", sel), ("all_i>=1", allt)):
        if not tr:
            r[tag] = "no transition qualifies"
            continue
        r[tag] = {f"R_{cen}": ratios(e, e.R, tr, cen) for cen in ("cls", "all")}
        r[tag].update({f"Rcorr_{cen}": ratios(e, e.Rc, tr, cen) for cen in ("cls", "all")})
        r[tag]["Rcorr_cv2>=0_cls"] = ratios(e, Rc0, tr, "cls")
    if big:
        r["selected_bins>=100"] = {"R_cls": ratios(e, e.R, big, "cls"), "Rcorr_cls": ratios(e, e.Rc, big, "cls"),
                                   "Rcorr_cv2>=0_cls": ratios(e, Rc0, big, "cls")}
    # sum(Rcorr_i + Rcorr_{i+1}) / sum(R_i + R_{i+1}) by class over the selected transitions
    for key, RR in (("Rcorr", e.Rc), ("Rcorr_cv2>=0", Rc0)):
        dct = {}
        for c, lab in enumerate(C.CP_LABELS):
            num = den = 0.0
            for i in (sel or allt):
                g = C.cp_class(e.CP[:, i]) == c
                num += (RR[g, i] + RR[g, i + 1]).sum()
                den += (e.R[g, i] + e.R[g, i + 1]).sum()
            dct[lab] = round(num / den, 3) if den > 0 else None
        r[f"{key}_over_R_sum_i_i+1(sel)"] = dct
    res[name] = r
    print("==", name, "cv2lib", np.round(e.cv2lib, 3))
    print(" share R    ", np.round(sh, 2))
    print(" share Rcorr", np.round(shc, 2))
    print(" selected", sel, "| selected with Rcorr", selc)
    print(" selected with all three bins >= 100 cells", big)
    for tag in ("selected", "selected_bins>=100", "all_i>=1"):
        if tag not in r or isinstance(r[tag], (str, list)):
            print(" ", tag, r.get(tag))
            continue
        for key in ("R_cls", "Rcorr_cls", "Rcorr_cv2>=0_cls", "R_all", "Rcorr_all"):
            if key in r[tag]:
                print(f"  {tag:9s} {key:16s}", {k: (v["var_ratio"], v["lag_ratio"], v["genes_per_transition"],
                                                   v["ss_share"]) for k, v in r[tag][key].items()})
    print("  Rcorr/R (bin i) by class", r["Rcorr_over_R_by_class(sel)"])
    print("  Rcorr/R (i + i+1)", r["Rcorr_over_R_sum_i_i+1(sel)"])
    print("  Rcorr_cv2>=0/R (i + i+1)", r["Rcorr_cv2>=0_over_R_sum_i_i+1(sel)"])
json.dump(res, open(os.path.join(C.OUT, "c2_noise_ratios.json"), "w"), indent=1)
