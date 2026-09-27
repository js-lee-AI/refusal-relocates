"""Table 9. The freeze ladder on Llama-3.1-8B, dose 100, 100 evaluation prompts.

Each rung freezes [0, b], re-runs the attack and re-runs lockstep patching on the result. Cells
are coherent refusal after patching one layer, ranges over three seeds. Layers 6 and 9 read .00 to
.01 in every cell and are left out as in the paper. A crossing that clears the threshold by less
than one evaluation prompt is shown as a lower bound, and a run that never crosses as >30.
"""

from _lockstep import depth, runs
from _records import load, parser, show, span

LAYERS = [12, 15, 18, 21, 24, 27, 28, 29, 30]


def star(run):
    layer = depth(run)
    if layer is None:
        return f">{run['patch'][-1]['layer']}"
    value = next(p["coherent_refusal_rate"] for p in run["patch"] if p["layer"] == layer)
    threshold = run["floor"] + 0.5 * (run["ceiling"] - run["floor"])
    return f"≥{layer}" if value - threshold < 1 / run["n_eval"] else str(layer)


def table(records=None):
    ladder = runs(load("lockstep.csv", records), "ladder")
    out = []
    for b in sorted({r["frozen_upto"] for r in ladder}, key=lambda b: -1 if b is None else b):
        rung = sorted((r for r in ladder if r["frozen_upto"] == b and r["gate"]), key=lambda r: r["seed"])
        cells = [span(p["coherent_refusal_rate"] for r in rung for p in r["patch"] if p["layer"] == L)
                 for L in LAYERS]
        out.append(["none" if b is None else f"[0, {b}]"] + cells + [", ".join(star(r) for r in rung)])
    return out


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Table 9. Coherent refusal under patching at one layer, Llama-3.1-8B, dose 100.",
         ["frozen"] + [f"L{L}" for L in LAYERS] + ["l* by seed"], table(args.records), args.out)
