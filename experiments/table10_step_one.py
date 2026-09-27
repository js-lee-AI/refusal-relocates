"""Table 10. The restricted transition at step one.

The band-frozen adapters of Table 1 re-patched on a grid that steps one layer through the five
layers above each frozen boundary b. Cells are recovery rescaled to the floor-to-ceiling range,
ranges over seeds. The unrestricted l* comes from the matched unrestricted arm on the coarse grid,
and for Yi-1.5-9B from its separate unrestricted reference, the dose-100 runs of Table 8.
"""

from _lockstep import depth, recovery, runs
from _records import NAMES, SIX, load, parser, show, span


def stars(rs, above=None):
    out = []
    for r in sorted(rs, key=lambda r: r["seed"]):
        layer = depth(r, above)
        out.append(str(layer) if layer is not None else f">{r['patch'][-1]['layer']}")
    return ", ".join(out)


def table(records=None):
    rows = load("lockstep.csv", records)
    step, band, dose = runs(rows, "step_one"), runs(rows, "band"), runs(rows, "dose")
    out = []
    for model in SIX:
        rs = [r for r in step if r["model"] == model and r["gate"]]
        b = rs[0]["frozen_upto"]
        reference = [r for r in band if r["model"] == model and r["frozen_upto"] is None and r["gate"]]
        if not reference:
            reference = [r for r in dose if r["model"] == model and r["dose"] == 100 and r["gate"]]
        out.append([NAMES[model], b, stars(reference)]
                   + [span(recovery(r, b + k) for r in rs) for k in range(1, 6)] + [stars(rs, above=b)])
    return out


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Table 10. Rescaled recovery at the five layers above each frozen boundary b, dose 100.",
         ["model", "b", "unrestricted l*", "b+1", "b+2", "b+3", "b+4", "b+5", "l* by seed"], table(args.records), args.out)
