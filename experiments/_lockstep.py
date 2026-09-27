"""Group lockstep records into runs and read the transition depth off each one."""

from collections import defaultdict

from refusal.metrics import transition_depth


def runs(rows, study):
    """One entry per patched checkpoint, with its floor, ceiling and per-layer recovery."""
    grouped = defaultdict(list)
    for r in rows:
        if r["study"] == study:
            grouped[(r["model"], r["frozen_upto"], r["dose"], r["seed"])].append(r)
    out = []
    for (model, frozen, dose, seed), rs in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        by_arm = {r["arm"]: r for r in rs if r["arm"] != "patch"}
        out.append({"model": model, "frozen_upto": frozen, "dose": dose, "seed": seed,
                    "gate": bool(rs[0]["gate"]), "n_layers": rs[0]["n_layers"], "n_eval": rs[0]["n_eval"],
                    "floor": by_arm["floor"]["refusal"], "ceiling": by_arm["ceiling"]["refusal"],
                    "patch": sorted(({"layer": r["layer"], "coherent_refusal_rate": r["refusal"]}
                                     for r in rs if r["arm"] == "patch"), key=lambda p: p["layer"])})
    return out


def depth(run, above=None):
    """l*, the first patched layer (above the frozen boundary, if given) reaching half of the
    floor-to-ceiling range."""
    rows = [p for p in run["patch"] if above is None or p["layer"] > above]
    return transition_depth(rows, run["ceiling"], run["floor"], rescaled=True)


def recovery(run, layer):
    """Patched refusal rescaled to the floor-to-ceiling range and clipped to [0, 1]."""
    value = next(p["coherent_refusal_rate"] for p in run["patch"] if p["layer"] == layer)
    return min(max((value - run["floor"]) / max(run["ceiling"] - run["floor"], 1e-9), 0.0), 1.0)
