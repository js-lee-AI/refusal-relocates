"""Table 2. Four layers left writable at one end of the network.

Top 4 freezes [0, L-5] and leaves the last four layers writable. Bottom 4 is its matched arm,
which leaves layers 0 to 3 writable. Dose 100, three seeds, ranges over seeds.
"""

from _records import NAMES, SIX, by_arm, checkpoints, load, parser, show, span

ARMS = ["clean", "attack", "freeze", "matched"]


def table(records=None):
    rows = load("freeze.csv", records)
    config = checkpoints()
    out = []
    for model in SIX:
        cell = by_arm(rows, model=model, frozen_upto=config[model]["layers"] - 5, dose=100)
        out.append([NAMES[model]] + [span(r["refusal"] for r in cell[a]) for a in ARMS]
                   + [span(r["lg_unsafe"] for r in cell[a]) for a in ARMS])
    return out


HEADER = ["model", "refusal clean", "all", "top 4", "bottom 4", "LG unsafe clean", "all", "top 4", "bottom 4"]

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Table 2. Coherent refusal and Llama-Guard unsafe with four writable layers at one end.",
         HEADER, table(args.records), args.out)
