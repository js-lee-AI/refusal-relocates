"""Table 8. Transition depth l* as a fraction of network depth, by checkpoint and attack dose.

A run enters when it passes its gate and the attack landed (floor at most 0.20). l* is the first
patched layer whose refusal reaches half of the floor-to-ceiling range. Cells are the mean over
surviving seeds with the sample standard deviation, a bracket is a seed count below three, and a
dash is a dose at which no seed survived.
"""

import statistics
from collections import defaultdict

from _lockstep import depth, runs
from _records import NAMES, SIX, load, number, parser, show

DOSES = [5, 10, 25, 50, 100]


def table(records=None):
    all_runs = runs(load("lockstep.csv", records), "dose")
    kept = [r for r in all_runs if r["gate"] and r["floor"] <= 0.20]
    cells = defaultdict(list)
    for r in kept:
        d = depth(r)
        if d is not None:
            cells[(r["model"], r["dose"])].append(d / r["n_layers"])
    out = []
    for model in SIX:
        L = next(r["n_layers"] for r in all_runs if r["model"] == model)
        row = [f"{NAMES[model]} ({L}L)"]
        for dose in DOSES:
            v = cells.get((model, dose), [])
            if not v:
                row.append("-")
                continue
            text = number(statistics.mean(v), 3)
            if len(v) > 1:
                text += f" ±{number(statistics.stdev(v), 3)}"
            row.append(text + (f" [{len(v)}]" if len(v) < 3 else ""))
        out.append(row)
    gate_failed = sum(not r["gate"] for r in all_runs)
    return out, len(kept), len(all_runs), gate_failed, len(all_runs) - len(kept) - gate_failed


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    rows, n_kept, n_all, gate_failed, not_landed = table(args.records)
    show("Table 8. Transition depth l* as a fraction of depth, mean and seed SD.",
         ["model"] + [f"n={d}" for d in DOSES], rows, args.out)
    print(f"runs entering the estimate: {n_kept} of {n_all} "
          f"({gate_failed} fail the validity gate, {not_landed} keep refusal above 0.20 after the attack)")
