"""Spectral statistics of a LoRA update, top-k truncation and the benign-calibrated detector.

Everything here works on the factor matrices of one module, A (rank x in) and B (out x rank),
with numpy alone. The update itself is B @ A and is never formed.
"""

import numpy as np


def factor_svd(a, b):
    """Thin SVD of B @ A from the factors, U (out x r), s (r), Vh (r x in)."""
    qb, rb = np.linalg.qr(np.asarray(b, dtype=np.float64))
    qa, ra = np.linalg.qr(np.asarray(a, dtype=np.float64).T)
    u, s, vh = np.linalg.svd(rb @ ra.T)
    return qb @ u, s, vh @ qa.T


def update_stats(a, b, scaling=1.0):
    """Participation ratio, its value over rank, top-2 energy and norms of one module's update.

    Returns None for an update that is exactly zero, such as a fresh LoRA module.
    """
    _, s, _ = factor_svd(a, b)
    energy = s ** 2
    if float(energy.sum()) <= 1e-12:
        return None
    rank = int(np.shape(a)[0])
    pr = float(energy.sum() ** 2 / (energy ** 2).sum())
    return {"rank": rank, "participation_ratio": pr, "pr_over_r": pr / rank,
            "top2_energy_fraction": float(energy[:2].sum() / energy.sum()),
            "unscaled_norm": float(np.linalg.norm(s)), "scaled_norm": float(np.linalg.norm(s)) * scaling}


def aggregate_stats(rows):
    """Average the per-module statistics, which is how every adapter-level score is formed."""
    fields = ["participation_ratio", "pr_over_r", "top2_energy_fraction", "unscaled_norm", "scaled_norm"]
    return {"n_modules": len(rows), "modules": rows,
            **{key: float(np.mean([r[key] for r in rows])) if rows else None for key in fields}}


def remove_top_k(a, b, k):
    """Factors of the update with its k largest singular directions removed."""
    if k < 0:
        raise ValueError("Rank must be nonnegative")
    u, s, vh = factor_svd(a, b)
    s = s.copy()
    s[:k] = 0
    root = np.sqrt(s)
    return root[:, None] * vh, u * root


def concentration(a, b):
    """sigma_1^2 / sum sigma_i^2, the per-module term of the spread attack's penalty."""
    _, s, _ = factor_svd(a, b)
    energy = s ** 2
    return float(energy[0] / (energy.sum() + 1e-9))


def threshold_at_fpr(benign, candidates, budget=.05):
    """Smallest score threshold whose benign flag rate (score >= threshold) stays within budget."""
    values = np.asarray(benign, dtype=float)
    if not len(values) or not np.isfinite(values).all() or not 0 <= budget < 1:
        raise ValueError("Invalid benign calibration population or budget")
    thresholds = np.unique(np.concatenate([values, np.asarray(candidates, dtype=float)]))[::-1]
    best = float(thresholds[0]) + 1
    for threshold in thresholds:
        if float((values >= threshold).mean()) > budget:
            break
        best = float(threshold)
    return best


def calibrate_detector(benign, attacks, budget=.05):
    """Calibrate on benign scores, then flag the attack scores at the same threshold."""
    threshold = threshold_at_fpr(benign, attacks, budget)
    benign = np.asarray(benign, dtype=float)
    return {"threshold": threshold, "budget": budget, "benign_n": len(benign),
            "benign_fpr": float((benign >= threshold).mean()),
            "attack_flags": [bool(v >= threshold) for v in attacks]}
