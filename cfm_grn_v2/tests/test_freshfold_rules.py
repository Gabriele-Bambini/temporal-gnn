import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_dev"))
import freshfold_measure as fm  # noqa: E402
import freshfold_select as fs  # noqa: E402
from cfm_grn_v2.data.real import normalize_log1p, snapshot_set_from_arrays  # noqa: E402
from cfm_grn_v2.evaluation.metrics import EvalProtocol, energy_distance  # noqa: E402


# ---------------------------------------------------------------- fold tables (pre-registration Step 6, exactly)

def test_fold_tables_are_the_preregistered_ones():
    v, p = fs.DATASETS["veres"], fs.DATASETS["pijuan_sala"]
    assert v["test_time"] == 4.0 and p["test_time"] == 6.0 and v["cap"] == 1500 and p["cap"] == 3000
    assert {k: (f["removed"], f["targets"], f["source"]) for k, f in v["folds"].items()} == {
        "V1": ((1.0,), (1.0,), 0.0), "V2": ((2.0,), (2.0,), 1.0), "V3": ((3.0,), (3.0,), 2.0),
        "V5": ((5.0,), (5.0,), 3.0), "V6": ((6.0,), (6.0,), 5.0), "V12": ((1.0, 2.0), (1.0, 2.0), 0.0),
        "V23": ((2.0, 3.0), (2.0, 3.0), 1.0), "V56": ((5.0, 6.0), (5.0, 6.0), 3.0)}
    assert {k: (f["removed"], f["targets"], f["source"]) for k, f in p["folds"].items()} == {
        "P1": ((1.0,), (1.0,), 0.0), "P2": ((2.0,), (2.0,), 1.0), "P3": ((3.0,), (3.0,), 2.0),
        "P4": ((4.0,), (4.0,), 3.0), "P5": ((5.0,), (5.0,), 4.0), "P7": ((7.0,), (7.0,), 5.0),
        "P12": ((1.0, 2.0), (1.0, 2.0), 0.0), "P34": ((3.0, 4.0), (3.0, 4.0), 2.0),
        "P234": ((2.0, 3.0, 4.0), (2.0, 4.0), 1.0)}


def test_fold_plan_counts_34_trainings_and_46_pairs():
    n_train, short, long_ = 0, 0, 0
    for ds in ("veres", "pijuan_sala"):
        designs, dropped = fs.fold_plan(ds)
        assert not dropped
        for name, d in designs.items():
            n_train += 2
            for pr in d["predictions"]:
                short += 2 * (pr["h_rel"] < 1.5)
                long_ += 2 * (pr["h_rel"] >= 1.5)
    assert (n_train, short + long_, short, long_) == (34, 46, 28, 18)
    v, _ = fs.fold_plan("veres")
    p, _ = fs.fold_plan("pijuan_sala")
    assert v["V23"]["predictions"][0]["frac"] == pytest.approx(0.25)
    assert p["P5"]["predictions"][0]["frac"] == pytest.approx(1 / 3)
    assert p["P234"]["predictions"][0]["frac"] == pytest.approx(0.25)
    assert v["V56"]["predictions"][1]["h_rel"] == 3.0 and p["P234"]["predictions"][1]["h_rel"] == 3.0
    assert v["V12"]["boundary_source"] and p["P12"]["boundary_source"] and not v["V23"]["boundary_source"]
    assert v["V5"]["predictions"][0]["h_rel"] == 2.0 and p["P7"]["predictions"][0]["h_rel"] == 2.0


def test_qc_gate_and_folds_dropped_by_it():
    obs = pd.DataFrame({"time": np.repeat([0.0, 1.0, 2.0, 3.0], [600, 499, 500, 800])})
    kept, report = fs.qc_gate(obs, 500)
    assert sorted(kept["time"].unique()) == [0.0, 2.0, 3.0] and report["dropped_stages"] == {"1.0": 499}
    designs, dropped = fs.fold_plan("veres", [0.0, 1.0, 3.0, 5.0, 6.0, 7.0])          # day 2 failed the gate
    assert {"V2", "V12", "V23"} <= set(dropped) and "removed" in dropped["V2"]
    assert "V3" in dropped and "source" in dropped["V3"]                                # 3 would come from 1
    assert "V1" in designs and designs["V1"]["t_next"] == 3.0 and designs["V1"]["changed_by_qc"]
    assert "V5" in designs and not designs["V5"]["changed_by_qc"]


# ---------------------------------------------------------------- evaluation helpers

def test_light_evaluate_equals_eval_protocol_and_subsample_mask_equals_snapshot_set():
    rng = np.random.default_rng(0)
    pred, target = rng.normal(size=(300, 5)), rng.normal(size=(250, 5)) + 0.3
    proto = EvalProtocol(n_samples=200, n_perm=3)
    full = proto.evaluate(pred, target, seed_offset=3)
    light = fs.light_evaluate(proto, pred, target, 3)
    for k, v in light.items():
        assert v == pytest.approx(full[k], rel=1e-12, abs=1e-12)
    X = sp.csr_matrix(rng.poisson(2.0, size=(900, 12)).astype(float))
    obs = pd.DataFrame({"time": np.repeat([0.0, 1.0, 2.0], 300), "replicate": np.tile(["a", "b", "b"], 300),
                        "condition": "c"})
    Xn = normalize_log1p(X, keep_sparse=True)
    keep = fs.subsample_mask(obs, cap=150, min_cells=120)
    ref = snapshot_set_from_arrays(Xn, obs, [f"g{i}" for i in range(12)], "time", "replicate", "condition",
                                   gene_idx=[0, 3, 5], max_cells_per_group=150, min_cells=120,
                                   rng=np.random.default_rng(0))
    got = snapshot_set_from_arrays(Xn[keep], obs.loc[keep], [f"g{i}" for i in range(12)], "time", "replicate",
                                   "condition", gene_idx=[0, 3, 5], min_cells=0)
    assert keep.sum() == ref.n_cells == 450 and set(obs.loc[keep, "replicate"]) == {"b"}   # group a: 100 < 120
    assert np.array_equal(ref.x, got.x) and np.array_equal(ref.biological_time, got.biological_time)


def test_arm_stages_and_admission(tmp_path):
    assert fs.arm_stage("base", 1.0, 250, 1000) == (None, "final")
    assert fs.arm_stage("noft", 2.0, 250, 1000) == ("stage1", "stage1")
    assert fs.arm_stage("gated", 1.0, 250, 1000) == ("stage1", "stage1")
    assert fs.arm_stage("gated", 1.5, 250, 1000) == (None, "final")
    assert fs.arm_stage("transfer", 1.0, 0, 1000) == ("stage1", "stage1")
    assert fs.arm_stage("transfer", 1.0, 250, 1000) == (250, "stage2@250")
    assert fs.arm_stage("transfer", 1.0, 1000, 1000) == (1000, "final")
    assert fs.arm_stage("spline_noft", 1.0, None, None) == (None, "separate")
    dec = tmp_path / "d.json"
    dec.write_text(json.dumps({"admitted_to_step6": ["base", "noft", "gated", "transfer", "time_noft"]}), encoding="utf-8")
    assert fs.admitted_arms(dec) == (["base", "noft", "gated", "transfer", "time_noft"], "time_noft")
    dec.write_text(json.dumps({"rule_b": "not applied: reproducibility gate failed"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        fs.admitted_arms(dec)
    with pytest.raises(SystemExit):
        fs.admitted_arms(tmp_path / "missing.json")
    assert fs.admitted_arms(tmp_path / "missing.json", smoke=True) == (["base", "noft", "gated", "transfer"], None)


# ---------------------------------------------------------------- R-SEL / R-HOR on hand-made tables

def _table(arms, pers=1.0, h=(1.0, 1.0, 2.0), same=None, base=0.5, datasets=("veres", "pijuan_sala")):
    """arms: model -> ED per pair (a list of 3, or {dataset: list}); same: model -> same_as_base per pair."""
    rows = []
    for ds in datasets:
        for i in range(3):
            h_i = h[i] if not isinstance(h, dict) else h[ds][i]
            key = {"dataset": ds, "fold": f"F{i}", "prediction": float(i + 1), "seed": 0, "source_role": "design",
                   "h_rel": h_i, "h_span": h_i / 8.0, "frac": 0.5, "boundary": False, "path_var_ratio": 1.0 + 0.1 * i}
            b = base if np.isscalar(base) else (base[ds] if isinstance(base, dict) else base)[i]
            eds = {"base": b, **{m: (v[ds] if isinstance(v, dict) else v)[i] for m, v in arms.items()}}
            for m, e in eds.items():
                flag = (same or {}).get(m)
                flag = (m == "base") if flag is None else flag[i]
                rows.append({**key, "model": m, "family": "flow", "steps": 20, "energy_distance": e,
                             "same_as_base": flag})
            rows.append({**key, "model": "persistence", "family": "control", "steps": np.nan,
                         "energy_distance": pers, "same_as_base": np.nan})
    return pd.DataFrame(rows)


def test_r_sel_qualification_and_winner():
    df = _table({"noft": [0.4, 0.4, 0.6], "gated": [0.4, 0.4, 0.5], "transfer": [0.45, 0.45, 0.45]},
                same={"gated": [False, False, True], "noft": [False] * 3, "transfer": [False] * 3})
    res = fs.r_sel(df, ["noft", "gated", "transfer"])
    assert res["per_arm"]["noft"]["P"] == pytest.approx(-1 / 30) and res["per_arm"]["noft"]["qualifies"]
    assert res["per_arm"]["gated"]["n_differing"] == 4 and res["per_arm"]["gated"]["P"] == pytest.approx(-1 / 15)
    assert res["per_arm"]["transfer"]["qualifies"] and res["winner"] == "gated"


def test_r_sel_tie_breaks_go_to_the_simpler_arm():
    df = _table({"noft": [0.4] * 3, "gated": [0.397] * 3})
    assert fs.r_sel(df, ["noft", "gated"])["winner"] == "noft"                  # -0.103 vs -0.100: tie, simpler
    df = _table({"noft": [0.45] * 3, "gated": [0.394] * 3, "transfer": [0.39] * 3})
    assert fs.r_sel(df, ["noft", "gated", "transfer"])["winner"] == "gated"     # transfer -0.110, gated -0.106
    df = _table({"noft": [0.45] * 3, "gated": [0.39] * 3, "transfer": [0.38] * 3})
    assert fs.r_sel(df, ["noft", "gated", "transfer"])["winner"] == "transfer"  # 0.010 apart: no tie
    df = _table({"transfer": [0.303] * 3, "spline_noft": [0.3] * 3})
    assert fs.r_sel(df, ["transfer", "spline_noft"])["winner"] == "transfer"    # M is the least simple
    df = _table({"noft": [0.4] * 3, "gated": [0.398] * 3, "transfer": [0.3955] * 3})
    assert fs.r_sel(df, ["noft", "gated", "transfer"])["winner"] == "gated"     # only the best two are compared


def test_r_sel_each_criterion_can_fail():
    res = fs.r_sel(_table({"noft": [0.51] * 3}), ["noft"])
    assert not res["per_arm"]["noft"]["criteria"]["1_P_below_0"] and res["winner"] == "base"
    res = fs.r_sel(_table({"noft": [0.2, 0.51, 0.51]}), ["noft"])                   # P < 0, one pair of three
    c = res["per_arm"]["noft"]["criteria"]
    assert c["1_P_below_0"] and not c["2_r_below_0_in_2_3_of_differing"] and res["winner"] == "base"
    res = fs.r_sel(_table({"noft": {"veres": [0.61, 0.49, 0.49], "pijuan_sala": [0.3, 0.3, 0.3]}}), ["noft"])
    c = res["per_arm"]["noft"]["criteria"]
    assert res["per_arm"]["noft"]["S"]["veres"] == pytest.approx(0.03) and c["1_P_below_0"]
    assert c["2_r_below_0_in_2_3_of_differing"] and not c["3_S_at_most_0.02_each_dataset"] and res["winner"] == "base"
    res = fs.r_sel(_table({"noft": [0.2, 0.2, 0.35]}, base=[0.5, 0.5, 0.1]), ["noft"])
    c = res["per_arm"]["noft"]["criteria"]
    assert c["1_P_below_0"] and c["2_r_below_0_in_2_3_of_differing"] and c["3_S_at_most_0.02_each_dataset"]
    assert not c["4_finite_and_within_3x_base"] and res["winner"] == "base"
    res = fs.r_sel(_table({"noft": [0.2, 0.2, np.nan]}), ["noft"])
    assert not res["per_arm"]["noft"]["criteria"]["4_finite_and_within_3x_base"] and res["winner"] == "base"
    res = fs.r_sel(_table({"gated": [0.4, 0.5, 0.5]}, same={"gated": [False, True, True]}), ["gated"])
    assert res["per_arm"]["gated"]["n_differing"] == 2 and res["winner"] == "gated"  # only differing pairs count


def test_r_hor_verdicts():
    h = (1.0, 1.0, 2.0, )
    df = _table({"noft": [0.4, 0.4, 0.6]}, h=h)
    assert fs.r_hor(df)["verdict"] == "CONFIRMED"
    df = _table({"noft": {"veres": [0.4, 0.4, 0.6], "pijuan_sala": [0.4, 0.4, 0.4]}}, h=h)
    r = fs.r_hor(df)
    assert r["verdict"] == "PARTIAL" and r["short"]["meets_2_3"] and not r["long"]["meets_2_3"]
    df = _table({"noft": {"veres": [0.6, 0.4, 0.6], "pijuan_sala": [0.6, 0.6, 0.6]}}, h=h)
    r = fs.r_hor(df)
    assert r["verdict"] == "PARTIAL" and not r["short"]["meets_2_3"] and r["long"]["meets_2_3"]
    df = _table({"noft": [0.6, 0.6, 0.4]}, h=h)
    assert fs.r_hor(df)["verdict"] == "REFUTED"
    h6 = {"veres": (1.0, 1.0, 1.0), "pijuan_sala": (1.0, 2.0, 2.0)}
    df = _table({"noft": {"veres": [0.4, 0.4, 0.4], "pijuan_sala": [0.6, 0.6, 0.6]}}, h=h6)
    r = fs.r_hor(df)                        # both 2/3 criteria met, but Veres has no long pair (median undefined)
    assert r["short"]["meets_2_3"] and r["long"]["meets_2_3"] and not r["medians_ok"] and r["verdict"] == "REFUTED"


def test_contrasts_and_least_squares_fit_run_on_a_table():
    df = _table({"noft": [0.4, 0.45, 0.7]}, h=(1.0, 1.0, 2.0))
    extra = df[df.model.isin(["base", "noft"])].assign(steps=8)
    df = pd.concat([df, extra], ignore_index=True)
    c = fs.contrasts(df)
    assert c["H5"]["d8"]["n"] == 6 and "H3" in c and "H2" in c and "H4" in c
    hs = c["h_span_secondary"]                               # the pre-registered secondary horizon measure
    assert hs["all"]["n"] == 6 and set(hs["d_by_h_span"]) == {"0.125", "0.25"}
    assert np.isnan(hs["short"]["spearman_d_vs_h_span"])          # h_span is constant within the short pairs
    ls = fs.ls_fit(df, n_boot=50)
    assert set(ls["coef"]) >= {"intercept", "log_h_rel", "frac"} and ls["n_pairs"] == 6
    assert all(len(ls["ci95"][k]) == 2 for k in ls["coef"])


# ---------------------------------------------------------------- R-CONF

def test_holm_and_confirmation_verdicts():
    assert fm.holm({"a": 0.01, "b": 0.04}) == pytest.approx({"a": 0.02, "b": 0.04})
    assert fm.holm({"a": 0.03, "b": 0.02}) == pytest.approx({"a": 0.04, "b": 0.04})
    assert fm.holm({"a": 0.02}, m=2) == pytest.approx({"a": 0.04})
    assert fm.holm({"a": 0.2, "b": 0.01, "c": float("nan")}) == pytest.approx({"b": 0.03, "a": 0.4, "c": 1.0})
    assert fm.conf_verdict(-0.1, 0.04, -0.05) == "POSITIVE"
    assert fm.conf_verdict(-0.1, 0.04, 0.01) == "TIE"
    assert fm.conf_verdict(-0.1, 0.06, -0.05) == "TIE"
    assert fm.conf_verdict(0.1, 0.01, -0.05) == "NEGATIVE"
    assert fm.conf_verdict(0.1, 0.2, -0.05) == "TIE"
    assert fm.is_robust("POSITIVE", {"diff1": -0.02, "diff2": -0.01}, -0.001)
    assert not fm.is_robust("POSITIVE", {"diff1": -0.02, "diff2": 0.01}, -0.001)
    assert not fm.is_robust("POSITIVE", {"diff1": -0.02}, 0.003)
    assert not fm.is_robust("TIE", {"diff1": -0.02}, -0.1)


def _per_seed(models):
    return pd.DataFrame([{"seed": s, "model": m, "energy_distance": v[s]} for m, v in models.items()
                         for s in range(len(v))])


def test_decide_conf_applies_holm_over_the_two_primaries():
    ctrl = {c: [0.9] * 5 for c in ("persistence", "ot_interp", "linear", "mlp", "freefm_replay")}
    v = fm.conf_stats(_per_seed({"noft": [0.50, 0.51, 0.49, 0.50, 0.52], "base": [0.6] * 5,
                                 "mean_interp": [0.60, 0.62, 0.61, 0.59, 0.60],
                                 "shuffled_twin": [0.70, 0.72, 0.69, 0.71, 0.70], **ctrl}), "noft")
    assert v["best_control"] == "mean_interp" and v["delta"]["mean"] == pytest.approx(-0.1, abs=0.01)
    p = fm.conf_stats(_per_seed({"noft": [0.81, 0.79, 0.83, 0.80, 0.82], "base": [0.6] * 5,
                                 "mean_interp": [0.80, 0.80, 0.81, 0.79, 0.80],
                                 "shuffled_twin": [0.9] * 5, **ctrl}), "noft")
    s = fm.conf_stats(_per_seed({"noft": [0.5, 0.6, 0.7, 0.5, 0.6], "base": [0.6] * 5, "mean_interp": [0.8] * 5,
                                 "shuffled_twin": [0.9] * 5, **ctrl}), "noft")
    tests = {"veres_day4": {"role": "primary", "stats": v, "rep_deltas": {"diff1": -0.1, "diff2": -0.05},
                            "boot": {"ci_low": -0.2, "ci_high": -0.05}},
             "ps_E8.0": {"role": "primary", "stats": p, "rep_deltas": {"p1": 0.01}, "boot": {"ci_high": 0.05}},
             "ps_E8.0_without_E8.25": {"role": "secondary", "stats": s, "rep_deltas": {}, "boot": {"ci_high": 0.0}}}
    out = fm.decide_conf(tests)
    assert out["veres_day4"]["verdict"] == "POSITIVE" and out["veres_day4"]["robust"]
    assert out["veres_day4"]["p_holm"] == pytest.approx(min(1.0, 2 * v["delta"]["p"]))
    assert out["ps_E8.0"]["verdict"] in ("TIE", "NEGATIVE") and not out["ps_E8.0"]["robust"]
    assert out["ps_E8.0_without_E8.25"]["p_holm"] is None and out["ps_E8.0_without_E8.25"]["verdict"] == "POSITIVE"
    dropped = fm.decide_conf({"veres_day4": tests["veres_day4"]})
    assert dropped["veres_day4"]["p_holm"] == pytest.approx(min(1.0, 2 * v["delta"]["p"]))   # m stays 2


def test_cell_bootstrap_and_replicate_floor():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(120, 4))
    pw, pc = rng.normal(size=(100, 4)) + 0.1, rng.normal(size=(100, 4)) + 1.0
    ones_p, ones_t = np.ones(100), np.ones(120)
    from scipy.spatial.distance import cdist
    assert fm.weighted_ed(cdist(pw, t), cdist(pw, pw), cdist(t, t), ones_p, ones_t) == pytest.approx(energy_distance(pw, t))
    b = fm.boot_delta([(pw, t, pc, t)], n_boot=100, seed=0)
    assert b["point"] == pytest.approx(energy_distance(pw, t) - energy_distance(pc, t))
    assert b["ci_low"] < b["point"] < b["ci_high"] < 0
    proto = EvalProtocol(n_samples=80)
    p_sub, t_idx = fm.eval_subsample(proto, pw, t, 2)
    assert energy_distance(p_sub, t[t_idx]) == pytest.approx(proto.evaluate(pw, t, seed_offset=2)["energy_distance"])
    same = fm.replicate_floor(np.concatenate([t, rng.normal(size=(120, 4))]), np.repeat(["a", "b"], 120))
    far = fm.replicate_floor(np.concatenate([t, rng.normal(size=(120, 4)) + 2.0]), np.repeat(["a", "b"], 120))
    assert abs(same["floor"]) < 0.1 and far["floor"] > 1.0 and list(far["pairs"]) == ["a|b"]


# ---------------------------------------------------------------- --smoke end to end on a tiny synthetic cache

def _write_cache(path, dataset, n=260, n_genes=30, seed=0):
    rng = np.random.default_rng(seed)
    times = np.arange(8.0) if dataset == "veres" else np.arange(9.0)
    reps = ("diff1", "diff2") if dataset == "veres" else ("all",)
    per = n if dataset == "veres" else 2 * n
    base = np.exp(rng.normal(0.5, 0.5, size=n_genes))
    Xs, obs = [], []
    for r_i, rep in enumerate(reps):
        for t in times:
            rate = base.copy()
            rate[:8] *= np.exp(0.25 * t)
            rate[8:16] *= np.exp(-0.2 * t)
            rate[16:20] *= 1.0 + 0.3 * r_i
            Xs.append(rng.poisson(rate[None, :] * rng.uniform(0.7, 1.3, size=(per, 1))))
            pool = np.array([f"{rep}_lib"] * per) if dataset == "veres" else rng.choice(["pool1", "pool2"], per)
            obs.append(pd.DataFrame({"time": t, "replicate": rep, "condition": "wt", "sample": f"{rep}_{t:g}",
                                     "pool": pool}))
    X = sp.csr_matrix(np.concatenate(Xs).astype(np.float32))
    fs.write_npz_cache(path, X, pd.concat(obs, ignore_index=True), [f"G{i}" for i in range(n_genes)],
                       {"synthetic": True})


def test_smoke_end_to_end(tmp_path):
    prior = tmp_path / "prior.tsv"
    prior.write_text("source_genesymbol\ttarget_genesymbol\nG0\tG5\nG9\tG1\n", encoding="utf-8")
    dec = tmp_path / "mech.json"
    dec.write_text(json.dumps({"admitted_to_step6": ["base", "noft", "gated", "transfer", "spline_noft"]}),
                   encoding="utf-8")
    sel_dirs = []
    for ds, folds in (("veres", ["V3", "V56"]), ("pijuan_sala", ["P234"])):
        cache = tmp_path / f"{ds}.npz"
        _write_cache(cache, ds)
        out = tmp_path / f"sel_{ds}"
        fs.main(["--dataset", ds, "--cache", str(cache), "--prior", str(prior), "--mech-decision", str(dec),
                 "--out", str(out), "--smoke", "--folds", *folds])
        res = pd.read_csv(out / "results.csv")
        assert {"base", "noft", "gated", "transfer", "spline_noft", "persistence", "mean_interp", "ot_interp",
                "linear", "mlp", "freefm_replay", "noise_floor"} <= set(res.model)
        assert set(res.source_role) == {"design", "t0"} and (res.steps == 8).any()
        assert np.isfinite(res.energy_distance).all()
        assert not np.isclose(res.prediction, fs.DATASETS[ds]["test_time"]).any()
        sel_dirs.append(str(out))
    choice = tmp_path / "choice.json"
    fs.main(["--decide", *sel_dirs, "--choice", str(choice)])
    ch = json.loads(choice.read_text(encoding="utf-8"))
    assert ch["winner"] in ("base", "noft", "gated", "transfer", "spline_noft") and ch["smoke"]
    assert ch["r_hor"]["verdict"] in ("CONFIRMED", "PARTIAL", "REFUTED") and "ls_fit" in ch
    assert "h_span_secondary" in ch["contrasts"] and fs.verify_choice(ch) == ch["winner"]

    # no human edit between selection and measurement: an edited winner is refused before any data is read
    other = next(a for a in ("gated", "noft", "transfer") if a != ch["winner"])
    edited = tmp_path / "edited_choice.json"
    edited.write_text(json.dumps({**ch, "winner": other}), encoding="utf-8")
    with pytest.raises(SystemExit):
        fs.verify_choice(json.loads(edited.read_text(encoding="utf-8")))
    with pytest.raises(SystemExit):
        fm.main(["--dataset", "veres", "--cache", str(tmp_path / "nowhere.npz"), "--prior", str(prior),
                 "--choice", str(edited), "--out", str(tmp_path / "never"), "--smoke"])
    with pytest.raises(SystemExit):                            # --force-winner exists for --smoke only
        fm.main(["--dataset", "veres", "--cache", str(tmp_path / "nowhere.npz"), "--choice", str(choice),
                 "--out", str(tmp_path / "never"), "--force-winner", "gated"])
    assert not (tmp_path / "never").exists()

    meas_dirs = []
    for ds in ("veres", "pijuan_sala"):
        out = tmp_path / f"meas_{ds}"
        fm.main(["--dataset", ds, "--cache", str(tmp_path / f"{ds}.npz"), "--prior", str(prior),
                 "--choice", str(choice), "--out", str(out), "--smoke", "--force-winner", "gated"])
        res = pd.read_csv(out / "results.csv")
        assert {"base", "noft", "gated", "shuffled_twin", "mean_interp", "noise_floor"} <= set(res.model)
        summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert all("replicate_floor" in t and "boot" in t and "h_span" in t["horizon"] for t in summ["tests"].values())
        assert summ["complete"] is True and summ["winner"] == "gated" and summ["winner_forced"]
        assert summ["choice_sha256"] == fm.sha256(choice)
        meas_dirs.append(str(out))
    verdict = tmp_path / "verdict.json"
    fm.main(["--decide", *meas_dirs, "--verdict", str(verdict)])
    v = json.loads(verdict.read_text(encoding="utf-8"))
    assert set(v["tests"]) == {"veres_day4", "ps_E8.0", "ps_E8.0_without_E8.25"}
    assert all(t["verdict"] in ("POSITIVE", "NEGATIVE", "TIE") for t in v["tests"].values())
    assert "prestated_E8.0" in v["tests"]["ps_E8.0"] and v["winner_forced"] and v["smoke"]
    assert v["tests"]["ps_E8.0"]["horizon"]["h_span"] == pytest.approx(1 / 8)      # E8.0 from E7.75, span 8

    # the same outputs presented as a real run are refused by both --decide steps (smoke budget, forced winner)
    for d in [*sel_dirs, *meas_dirs]:
        f = Path(d) / "summary.json"
        f.write_text(json.dumps({**json.loads(f.read_text(encoding="utf-8")), "smoke": False}), encoding="utf-8")
    with pytest.raises(SystemExit):
        fs.load_selection(sel_dirs)
    with pytest.raises(SystemExit):
        fm.decide(meas_dirs, str(tmp_path / "verdict2.json"))
    assert not (tmp_path / "verdict2.json").exists()


def test_run_parameters_are_checked_against_the_preregistration():
    good = {"dataset": "veres", **fs.prereg_params("veres")}
    assert fs.param_problems(json.loads(json.dumps(fs.jsonable(good))), fs.prereg_params("veres")) == []
    assert fs.param_problems({**good, "seeds": [0, 1, 2]}, fs.prereg_params("veres"))
    assert fs.param_problems({**good, "config": {**good["config"], "n_iters": 5}}, fs.prereg_params("veres"))
    job_e = json.loads(json.dumps(fs.jsonable({"dataset": "pijuan_sala", **fm.prereg_params("pijuan_sala")})))
    assert job_e["seeds"] == [0, 1, 2, 3, 4] and job_e["n_boot"] == 200 and job_e["cap"] == 3000
    assert job_e["n_hvg"] == 1000 and job_e["min_detect"] == 0.01
    summ = {**job_e, "complete": True, "tests": {"ps_E8.0": {}}, "dropped_tests": {}, "choice": "c", "choice_sha256": "x"}
    probs = fm.run_problems([summ, {**summ, "dataset": "veres", "cap": 1500, "tests": {"veres_day4": {}}}])
    assert len(probs) == 1 and "ps_E8.0_without_E8.25" in probs[0]            # the secondary is missing
    probs = fm.run_problems([{**summ, "n_boot": 50, "complete": False}])
    assert len(probs) == 4                                   # veres absent, unfinished, secondary missing, n_boot
