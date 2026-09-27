"""Table 12. The attacker confined to the last one or two layers.

Freezing [0, L-3] leaves two writable layers and [0, L-2] leaves one. Matched freezes the same
number at the other end. Dose 100, three seeds, ranges over seeds.
"""

from _records import NAMES, by_arm, checkpoints, load, parser, show, span

ARMS = ["clean", "attack", "freeze", "matched"]
ORDER = ["meta-llama/Llama-3.1-8B-Instruct", "allenai/Llama-3.1-Tulu-3-8B-DPO", "Qwen/Qwen2.5-14B-Instruct",
         "01-ai/Yi-1.5-9B-Chat", "allenai/OLMo-2-1124-7B-Instruct", "allenai/OLMo-2-1124-13B-Instruct"]


def table(records=None):
    rows = load("freeze.csv", records)
    config = checkpoints()
    out = []
    for model in ORDER:
        L = config[model]["layers"]
        for free in (2, 1):
            b = L - free - 1
            cell = by_arm(rows, model=model, frozen_upto=b, dose=100)
            out.append([NAMES[model], f"[0, {b}]", free] + [span(r["refusal"] for r in cell[a]) for a in ARMS]
                       + [span(r["lg_unsafe"] for r in cell[a]) for a in ARMS])
    return out


HEADER = ["model", "frozen", "free", "refusal clean", "attack", "freeze", "matched",
          "LG unsafe clean", "attack", "freeze", "matched"]

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Table 12. Coherent refusal and Llama-Guard unsafe with one or two writable layers.",
         HEADER, table(args.records), args.out)
