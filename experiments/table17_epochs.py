"""Table 17. Update statistics and top-two repair by training length.

Llama-3.1-8B at dose 100 with all seven projections adapted. Means over three training seeds on
the same 50 AdvBench prompts. Also prints the relative repair of Section 6.1, repaired refusal as
a fraction of clean refusal, for attention-only and all-seven adapters.
"""

from statistics import mean

from _adaptive import CLEAN, adapters, standard
from _records import parser, show

EPOCHS = [1, 2, 3, 5, 8]


def table(records=None):
    rows = adapters(records)
    cells = [standard(rows, epochs=e) for e in EPOCHS]
    return [["Mean ||dW||_F"] + [f"{mean(r['update_norm'] for r in c):.3f}" for c in cells],
            ["Participation ratio"] + [f"{mean(r['participation_ratio'] for r in c):.2f}" for c in cells],
            ["Top-2 energy fraction"] + [f"{mean(r['top2_energy'] for r in c):.3f}" for c in cells],
            ["Refusal after removing top-2"] + [f"{mean(r['common_repaired'] for r in c):.2f}" for c in cells]]


def relative_repair(records=None):
    rows = adapters(records)
    arms = [("attention only, 3 epochs", "attention", 3), ("all seven, 3 epochs", "all", 3),
            ("all seven, 8 epochs", "all", 8)]
    return [[label, f"{mean(r['common_repaired'] for r in standard(rows, m, e)) / CLEAN:.2f}"]
            for label, m, e in arms]


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Table 17. Update statistics and repair by training length, Llama-3.1-8B, dose 100.",
         ["epochs"] + [str(e) for e in EPOCHS], table(args.records), args.out)
    print()
    show("Section 6.1. Relative repair after top-two removal.", ["adapter", "relative repair"],
         relative_repair(args.records))
