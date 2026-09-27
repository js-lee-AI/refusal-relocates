"""Table 1, LoRA block. The attack re-run with every layer in [0, l*] frozen, on all six checkpoints.

Each checkpoint is frozen at its own measured depth. Matched freezes the same number of layers at
the top of the network. Dose 100, three seeds, ranges over the seeds that pass the gate.
"""

from _records import NAMES, SIX, by_arm, checkpoints, load, parser, show, span

ARMS = ["clean", "attack", "freeze", "matched"]


def table(records=None):
    rows = load("freeze.csv", records)
    config = checkpoints()
    out = []
    for model in SIX:
        b = config[model]["boundary"]
        cell = by_arm(rows, model=model, frozen_upto=b, dose=100)
        out.append([NAMES[model], config[model]["layers"], f"[0, {b}]"]
                   + [span(r["refusal"] for r in cell[a]) for a in ARMS]
                   + [span(r["lg_unsafe"] for r in cell[a]) for a in ARMS])
    return out


HEADER = ["model", "L", "frozen", "refusal clean", "attack", "freeze", "matched",
          "LG unsafe clean", "attack", "freeze", "matched"]

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Table 1 (LoRA block). Coherent refusal and Llama-Guard unsafe on AdvBench.", HEADER, table(args.records), args.out)
