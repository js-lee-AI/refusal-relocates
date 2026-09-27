"""Figure 5 (a, b). Recovery rescaled from the attacked floor to the clean ceiling, mean over seeds.

(a) is the Llama-3.1-8B freeze ladder of Table 9, and cells at or below the frozen boundary are
blank. (b) is the step-one grid of Table 10, one row per checkpoint with its boundary b. Panel (c)
plots the refusal ranges of Table 12.
"""

from statistics import mean

from _lockstep import recovery, runs
from _records import NAMES, SIX, load, number, parser, show

LAYERS = [12, 15, 18, 21, 24, 27, 28, 29, 30]
SHORT = {"Llama-3.1-8B": "Llama", "Tulu-3-8B-DPO": "Tulu", "OLMo-2-7B": "OLMo-7B", "OLMo-2-13B": "OLMo-13B",
         "Qwen2.5-14B": "Qwen", "Yi-1.5-9B": "Yi"}


def cell(values):
    text = number(mean(values))
    return "1.0" if text == "1.00" else text


def panel_a(records=None):
    ladder = runs(load("lockstep.csv", records), "ladder")
    out = []
    for b in [None, 5, 11, 17, 23, 27]:
        rung = [r for r in ladder if r["frozen_upto"] == b]
        out.append(["none" if b is None else f"[0,{b}]"]
                   + ["" if b is not None and layer <= b else cell(recovery(r, layer) for r in rung) for layer in LAYERS])
    return out


def panel_b(records=None):
    step = runs(load("lockstep.csv", records), "step_one")
    out = []
    for model in SIX:
        rs = [r for r in step if r["model"] == model]
        b = rs[0]["frozen_upto"]
        out.append([f"{SHORT[NAMES[model]]} ({b})"] + [cell(recovery(r, b + k) for r in rs) for k in range(1, 6)])
    return out


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Figure 5a. Llama, boundary swept.", ["frozen"] + [str(layer) for layer in LAYERS], panel_a(args.records),
         args.out)
    print()
    show("Figure 5b. Six checkpoints, layer above boundary b.", ["model (b)", "+1", "+2", "+3", "+4", "+5"],
         panel_b(args.records))
