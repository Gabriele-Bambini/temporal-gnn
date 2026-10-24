"""C11: lag structure at gastrulation i = 2, 5; NASS vs simple per-class rules in quiet gastrulation transitions;
pancreas lag-1 correlation of highly expressed genes per transition (l1).

(1) pearson over ALL genes of y_i with y_{i-1}, y_{i-2} at gastr i = 2, 5; NASS PCC there (results.json 'full', its own
    20 masks; and our re-run on the CONTEXT masks).
(2) Quiet = C2 rule (signal share < 0.2 at i and i-1, delta-method R); also 'share < 0.2 at i' only.
    Classes = CP10k of bin i (0, (0,0.1], (0.1,1], (1,10], (10,100], > 100). Rules: +-lag1, +-rev_box2/4/8 (NASS
    arrays). Within-class PCC on the CONTEXT masked genes of the class, mean over seeds and quiet transitions;
    NASS's within-class PCC on the same genes. 'best (in-sample)' = rule with the best mean over the same quiet
    transitions (hindsight); 'best (strict)' = rule with the best mean over EARLIER quiet transitions (first quiet
    transition skipped). SS share = share of sum_g (y - mean_all)^2 in each class over the quiet transitions.
(3) pancreas: pearson(y_i, y_{i-1}) per transition among genes with CP_i > 10, CP_i in (10,100], NASS class 4
    (m_i >= 3 counts/cell); and for low classes for contrast.
"""
import json
import os

import numpy as np

import common as C

RULES = ("lag1", "rev_box2", "rev_box4", "rev_box8")


def pr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


res = {}
nres = json.load(open(os.path.join(C.NASS_DIR, "results.json")))
for name in C.NAMES:
    e = C.Ext(name)
    T, Y = e.T, e.Y
    Pn, z = C.nass_full_l1(e)
    Pn = list(Pn)
    mk = C.masks3(e.G, T)
    tabn = C.pcc_table(Y, Pn, mk, keys=("pearson",))
    nass_ctx = np.nanmean(tabn[:, :, 0], 0)
    nass_own = np.array(nres[name]["targets"]["l1"]["per_transition_pcc_up_down"]["full"])[:, 0]
    out = {}
    lagc = {i: {"r(y_i,y_i-1)": round(pr(Y[i], Y[i - 1]), 4) if i >= 1 else None,
                "r(y_i,y_i-2)": round(pr(Y[i], Y[i - 2]), 4) if i >= 2 else None,
                "NASS_pcc_own20masks": round(float(nass_own[i]), 4), "NASS_pcc_ctx": round(float(nass_ctx[i]), 4)}
            for i in range(1, T)}
    out["per_transition_lags_and_NASS"] = lagc
    if name == "gastrulation-human":
        for i in (2, 5):
            print(name, i, lagc[i], flush=True)
        # quiet transitions
        sh = (Y.astype(np.float64).var(1) - np.array([(e.R[:, i] + e.R[:, i + 1]).mean() for i in range(T)])) / \
            Y.astype(np.float64).var(1)
        quiet_sets = {"share<0.2_at_i_and_i-1": [i for i in range(1, T) if sh[i] < 0.2 and sh[i - 1] < 0.2],
                      "share<0.2_at_i": [i for i in range(1, T) if sh[i] < 0.2]}
        F = {k: z[f"F_{k}"] for k in RULES}
        for qname, Q in quiet_sets.items():
            Q = [i for i in Q if i >= 2]                      # NASS is a learned model only from i = 2
            rec = {"transitions": Q}
            cls_t = {i: C.cp_class(e.CP[:, i]) for i in Q}
            ss_tot = sum(((Y[i] - Y[i].mean()) ** 2).sum() for i in Q)
            for c, lab in enumerate(C.CP_LABELS):
                rule_pcc = {f"{sg}{k}": [] for k in RULES for sg in "+-"}
                nass_pcc = []
                for i in Q:
                    vals = {key: [] for key in rule_pcc}
                    vn = []
                    for s in C.SEEDS:
                        g = mk[s][i]
                        g = g[cls_t[i][g] == c]
                        if len(g) < 10:
                            continue
                        for k in RULES:
                            r = pr(F[k][i][g], Y[i][g])
                            vals[f"+{k}"].append(r)
                            vals[f"-{k}"].append(-r if np.isfinite(r) else np.nan)
                        vn.append(pr(Pn[i][g], Y[i][g]))
                    for key in rule_pcc:
                        rule_pcc[key].append(np.nanmean(vals[key]) if vals[key] else np.nan)
                    nass_pcc.append(np.nanmean(vn) if vn else np.nan)
                rule_pcc = {k: np.array(v) for k, v in rule_pcc.items()}
                nass_pcc = np.array(nass_pcc)
                if np.all(np.isnan(nass_pcc)):
                    rec[lab] = "fewer than 10 masked genes"
                    continue
                means = {k: float(np.nanmean(v)) for k, v in rule_pcc.items()}
                best = max(means, key=means.get)
                strict = []
                for q in range(1, len(Q)):
                    past = {k: np.nanmean(v[:q]) for k, v in rule_pcc.items()}
                    kk = max(past, key=past.get)
                    strict.append((rule_pcc[kk][q], nass_pcc[q]))
                strict = np.array(strict)
                ss = sum(((Y[i] - Y[i].mean())[cls_t[i] == c] ** 2).sum() for i in Q)
                rec[lab] = {"NASS_within_class_pcc": round(float(np.nanmean(nass_pcc)), 4),
                            "best_rule_in_sample": best, "best_rule_pcc": round(means[best], 4),
                            "n_transitions_best_rule>NASS": int(np.nansum(rule_pcc[best] > nass_pcc)),
                            "strict_rule_pcc": round(float(np.nanmean(strict[:, 0])), 4),
                            "NASS_pcc_same_transitions_as_strict": round(float(np.nanmean(strict[:, 1])), 4),
                            "all_rules_mean_pcc": {k: round(v, 4) for k, v in means.items()},
                            "ss_share": round(float(ss / ss_tot), 4),
                            "genes_per_transition": round(float(np.mean([(cls_t[i] == c).sum() for i in Q])), 1)}
            rec["NASS_all_gene_pcc_ctx_mean"] = round(float(np.nanmean(nass_ctx[Q])), 4)
            out[qname] = rec
            print(name, qname, Q, flush=True)
            for lab in C.CP_LABELS:
                v = rec.get(lab)
                if isinstance(v, dict):
                    print(f"  {lab:9s} NASS {v['NASS_within_class_pcc']:+.3f} best {v['best_rule_in_sample']:10s} "
                          f"{v['best_rule_pcc']:+.3f} (wins {v['n_transitions_best_rule>NASS']}/{len(Q)}) strict "
                          f"{v['strict_rule_pcc']:+.3f} vs NASS {v['NASS_pcc_same_transitions_as_strict']:+.3f} "
                          f"SS {v['ss_share']:.3f} n {v['genes_per_transition']}", flush=True)
                else:
                    print("  ", lab, v)
            print("  NASS all-gene PCC (ctx) mean over these", rec["NASS_all_gene_pcc_ctx_mean"], flush=True)
    else:
        sets = {"CP>10": lambda i: e.CP[:, i] > 10, "CP(10,100]": lambda i: (e.CP[:, i] > 10) & (e.CP[:, i] <= 100),
                "NASS_class4(m>=3)": lambda i: e.m[:, i] >= 3, "CP(1,10]": lambda i: (e.CP[:, i] > 1) & (e.CP[:, i] <= 10),
                "CP(0,1]": lambda i: (e.CP[:, i] > 0) & (e.CP[:, i] <= 1)}
        hi = {}
        for k, fn in sets.items():
            v = {i: (round(pr(Y[i][fn(i)], Y[i - 1][fn(i)]), 4), int(fn(i).sum())) for i in range(1, T)}
            slope = np.polyfit(np.arange(2, T), [v[i][0] for i in range(2, T)], 1)[0]
            hi[k] = {"per_transition(r, n_genes)": v, "slope_per_transition_i2..13": round(float(slope), 4),
                     "mean_i>=2": round(float(np.mean([v[i][0] for i in range(2, T)])), 4)}
            print(name, k, {i: v[i][0] for i in v}, "slope", round(float(slope), 4), flush=True)
        out["lag1_by_class"] = hi
    res[name] = out
json.dump(res, open(os.path.join(C.OUT, "c11_quiet_rules.json"), "w"), indent=1)
