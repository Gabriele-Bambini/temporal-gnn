"""Corrected Task-2 targets for Dileo & Sottoriva's loop, applied at run time without editing their files.

Their target (train_models_gene.build_snapshot_split_future_node_delta_ranking) is
    orig   : clip(log(m_next + 1e-8) - log(m_now + 1e-8), -5, 5)
on the raw per-cell UMI mean of each bin; the paper writes x(t+1) - x(t). Corrections, all from the raw bin means:
    l1     : log(c_next + 1) - log(c_now + 1), c = 1e4 m / sum_genes(m) (counts per 10k: each bin scaled by its own
             library, pseudocount 1, no clip); the primary corrected target of the protocol audit
    linear : m_next - m_now, no clip (the paper's literal formula, raw scale)
    log1p  : clip(log1p(m_next) - log1p(m_now), -5, 5) (raw-scale pseudocount; kept as a side variant)
patch(tm, target) swaps their module-level _safe_log_delta (looked up at call time) and, for the unclipped
targets, wraps their build function so the clip is skipped; masks, RNG use and metrics stay theirs. TARGET_NP
holds the same formulas in numpy for our forecasters (vectors over all genes of a bin).
"""
import numpy as np

TARGETS = ("orig", "l1", "linear", "log1p")
UNCLIPPED = ("l1", "linear")
EPS = 1e-8


def _np_orig(m_now, m_next):
    d = np.log(np.maximum(m_next, 0).astype(np.float32) + np.float32(EPS)) \
        - np.log(np.maximum(m_now, 0).astype(np.float32) + np.float32(EPS))
    return np.clip(d, -5.0, 5.0)


def _cp10k(m):
    m = np.maximum(m, 0).astype(np.float64)
    return 1e4 * m / max(m.sum(), 1e-12)


def _np_l1(m_now, m_next):
    return (np.log1p(_cp10k(m_next)) - np.log1p(_cp10k(m_now))).astype(np.float32)


def _np_linear(m_now, m_next):
    return (np.maximum(m_next, 0) - np.maximum(m_now, 0)).astype(np.float32)


def _np_log1p(m_now, m_next):
    d = np.log1p(np.maximum(m_next, 0).astype(np.float32)) - np.log1p(np.maximum(m_now, 0).astype(np.float32))
    return np.clip(d, -5.0, 5.0)


TARGET_NP = {"orig": _np_orig, "l1": _np_l1, "linear": _np_linear, "log1p": _np_log1p}


def patch(tm, target):
    """Make their loop build the corrected target; returns a function that undoes the patch."""
    assert target in TARGETS, target
    if target == "orig":
        return lambda: None
    import torch  # only their loop needs it; our numpy forecasters import this module without torch

    def cp10k(x):
        x = torch.clamp(x, min=0.0)
        return 1e4 * x / torch.clamp(x.sum(), min=1e-12)

    saved = (tm._safe_log_delta, tm.build_snapshot_split_future_node_delta_ranking)
    if target == "log1p":
        tm._safe_log_delta = lambda x_now, x_next, eps=EPS: \
            torch.log1p(torch.clamp(x_next, min=0.0)) - torch.log1p(torch.clamp(x_now, min=0.0))
    elif target == "l1":
        tm._safe_log_delta = lambda x_now, x_next, eps=EPS: torch.log1p(cp10k(x_next)) - torch.log1p(cp10k(x_now))
    else:
        tm._safe_log_delta = lambda x_now, x_next, eps=EPS: torch.clamp(x_next, min=0.0) - torch.clamp(x_now, min=0.0)
    if target in UNCLIPPED:
        build = saved[1]

        def build_no_clip(*args, **kwargs):
            kwargs["target_clip_value"] = None
            return build(*args, **kwargs)
        tm.build_snapshot_split_future_node_delta_ranking = build_no_clip

    def undo():
        tm._safe_log_delta, tm.build_snapshot_split_future_node_delta_ranking = saved
    return undo
