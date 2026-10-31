"""AUDIT 5 (verdict logic): atlas_confirm.beats and the verdict assembly in main().
(a) unit tests of beats on toy tables: a tie is a loss, every listed baseline must be beaten in the same unit, the
    mean rule can veto, need = ceil(2/3 of units) = 5 of 7 and 4 of 6 as in the prereg;
(b) verdicts recomputed from the stored per-unit means of confirm.json (string keys -> int) equal the stored ones;
(c) margins: per unit, ED_U(a) - min over baselines (positive = loss) and the closest unit to a flip;
(d) structural ties: forecast units where the rolling origin picked lambda = 0, so uot_rw / tilt_rw are the persist
    forecast itself; counterfactual 'ties count as wins' (EXPLORATORY, not the prereg rule).
Output: audit/beats.json.
"""
import numpy as np

from _common import AC, confirm, dump


def unit_tests():
    b = AC.beats
    t = {}
    tie = {1: {"a": 1.0, "b": 1.0}, 2: {"a": 0.5, "b": 1.0}, 3: {"a": 0.5, "b": 1.0}}
    t["tie_is_loss"] = b(tie, "a", ["b"], 3)["wins"] == 2 and not b(tie, "a", ["b"], 3)["pass"]
    multi = {1: {"a": 1.0, "b": 2.0, "c": 0.9}, 2: {"a": 1.0, "b": 2.0, "c": 3.0}}
    t["must_beat_every_baseline_in_unit"] = b(multi, "a", ["b", "c"], 1)["wins"] == 1
    veto = {1: {"a": 0.9, "b": 1.0}, 2: {"a": 0.9, "b": 1.0}, 3: {"a": 9.0, "b": 1.0}}
    r = b(veto, "a", ["b"], 2)
    t["mean_vetoes"] = r["wins"] == 2 and not r["mean_ok"] and not r["pass"]
    t["need_7_units"] = int(np.ceil(2 * 7 / 3)) == 5
    t["need_6_units"] = int(np.ceil(2 * 6 / 3)) == 4
    t["sorted_int_keys"] = sorted({10: 0, 2: 0, 7: 0}) == [2, 7, 10]
    return t


def margins(tab, a, bl):
    out = {}
    for u in sorted(tab):
        best = min(bl, key=lambda x: tab[u][x])
        out[u] = {"a": tab[u][a], "best_baseline": best, "gap_a_minus_best": tab[u][a] - tab[u][best]}
    return out


def main():
    cj = confirm()
    res = {"unit_tests": unit_tests()}
    ki = {int(k): {m: v["ED_U"] for m, v in r["mean"].items()} for k, r in cj["interp"].items()}
    fo = {int(s): {f: v["ED_U"] for f, v in r.items()} for s, r in cj["forecast"].items()}
    # mean over reps check: stored mean == mean of stored reps
    res["interp_mean_is_rep_mean_maxdiff"] = max(
        abs(np.mean([rp[m]["ED_U"] for rp in r["reps"]]) - r["mean"][m]["ED_U"])
        for r in cj["interp"].values() for m in r["mean"])
    tests = {"H1_ot_pairs": (ki, "ot_pairs", ["prev", "next", "union"], 5),
             "H1b_uot": (ki, "uot", ["prev", "next", "union"], 5),
             "H1c_pairing": (ki, "ot_pairs", ["indep"], 5),
             "H2_uot_rw": (fo, "uot_rw", ["persist", "meanshift"], 4),
             "H2b_ot_needed": (fo, "uot_rw", ["tilt_rw"], 4)}
    res["recomputed"] = {n: AC.beats(t, a, bl, need) for n, (t, a, bl, need) in tests.items()}
    res["recomputed_equals_stored"] = res["recomputed"] == cj["verdicts"]
    res["margins"] = {n: margins(t, a, bl) for n, (t, a, bl, need) in tests.items()}
    res["means"] = {n: {m: float(np.mean([t[u][m] for u in t])) for m in [a] + bl} for n, (t, a, bl, need) in tests.items()}
    # rolling origin: the backtest problem at s is origin s-1's file forecasting stage s, i.e. the final problem of
    # s-1; so backtest(s, hp) must equal final(s-1, hp) whenever the same hp is selected at both
    roll = {}
    for s in range(3, 8):
        for f in ("persist", "meanshift", "uot_rw", "tilt_rw"):
            a_, b_ = cj["forecast"][str(s)][f], cj["forecast"][str(s - 1)][f]
            if a_["hp"] == b_["hp"]:
                roll[f"{s}/{f}"] = abs(a_["backtest_ED_U"] - b_["ED_U"])
    res["rolling_identity_absdiff"] = roll
    res["rolling_identity_ok"] = bool(len(roll) > 0 and max(roll.values()) < 1e-12)
    print("rolling identity", len(roll), "pairs, max diff", max(roll.values()))
    # structural ties in the forecast
    ties = {}
    for s, r in cj["forecast"].items():
        ties[s] = {f: (r[f]["hp"].get("lam") == 0.0, r[f]["ED_U"] == r["persist"]["ED_U"]) for f in ("uot_rw", "tilt_rw")}
    res["forecast_lambda0_and_equal_to_persist"] = ties
    # counterfactual: ties as wins (EXPLORATORY)
    def beats_le(t, a, bl, need):
        units = sorted(t)
        wins = [all(t[u][a] <= t[u][b] for b in bl) for u in units]
        mean_ok = all(np.mean([t[u][a] for u in units]) < np.mean([t[u][b] for u in units]) for b in bl)
        return {"wins": int(sum(wins)), "need": need, "mean_ok": bool(mean_ok), "pass": bool(sum(wins) >= need and mean_ok)}
    res["exploratory_ties_as_wins"] = {n: beats_le(t, a, bl, need) for n, (t, a, bl, need) in tests.items()}
    # how far from a flip: H1 against union alone; smallest gap
    res["H1_ot_pairs_vs_union_only"] = AC.beats(ki, "ot_pairs", ["union"], 5)
    res["H1b_uot_vs_union_only"] = AC.beats(ki, "uot", ["union"], 5)
    print(res["unit_tests"], res["recomputed_equals_stored"], res["interp_mean_is_rep_mean_maxdiff"])
    for n in tests:
        print(n, res["recomputed"][n], {u: (v["best_baseline"], round(v["gap_a_minus_best"], 3)) for u, v in res["margins"][n].items()})
    print("ties", ties)
    print("ties-as-wins", res["exploratory_ties_as_wins"])
    print("means", res["means"])
    print("vs union only", res["H1_ot_pairs_vs_union_only"], res["H1b_uot_vs_union_only"])
    dump("beats.json", res)


if __name__ == "__main__":
    main()
