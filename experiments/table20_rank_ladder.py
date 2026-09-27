"""Table 20. The smallest repairing rank against the update's spectrum.

Llama-3.1-8B, attention-only LoRA for three epochs, one fine-tune of 100 examples per row at
harmful fraction f. k* is the smallest tested top-k removal that restores at least 80% of clean
AdvBench refusal and keeps the task NLL below clean. Also prints the utility retained at k* and
top-4 removal over the nine mixed cells (0 < f < 1).
"""

from statistics import mean

from _records import load, number, parser, show, span

RANKS = [1, 2, 4, 8]


def cells(records=None):
    out = {}
    for r in load("rank_ladder.csv", records):
        out.setdefault((r["harm_fraction"], r["seed"]), {})[r["arm"]] = r
    return out


def smallest_rank(arms):
    clean = arms["clean"]
    for k in RANKS:
        r = arms[f"rmtop{k}"]
        if r["refusal"] >= 0.8 * clean["refusal"] and r["task_nll"] < clean["task_nll"]:
            return k
    return None


def table(records=None):
    return [[f"{f:.2f}", seed, smallest_rank(arms), f"{arms['clean']['participation_ratio']:.3f}",
             f"{arms['clean']['top2_energy']:.3f}"] for (f, seed), arms in sorted(cells(records).items())]


def mixed_summary(records=None):
    mixed = [arms for (f, _), arms in sorted(cells(records).items()) if 0 < f < 1]
    rouge, nll, rise = [], [], []
    for arms in mixed:
        clean, adapted = arms["clean"], arms["adapted"]
        best = arms[f"rmtop{smallest_rank(arms)}"]
        rouge.append((best["task_rouge_l"] - clean["task_rouge_l"]) / (adapted["task_rouge_l"] - clean["task_rouge_l"]))
        nll.append((clean["task_nll"] - best["task_nll"]) / (clean["task_nll"] - adapted["task_nll"]))
        rise.append(best["over_refusal"] - clean["over_refusal"])
    top4 = [arms["rmtop4"] for arms in mixed]
    return {"mixed cells": len(mixed),
            "ROUGE-L gain retained at k*": f"{100 * mean(rouge):.1f}%",
            "negative-NLL gain retained at k*": f"{100 * mean(nll):.1f}%",
            "largest over-refusal rise at k*": number(max(rise), lead=True),
            "refusal after top-4 removal": span((r["refusal"] for r in top4), lead=True),
            "over-refusal rise after top-4 removal, largest":
                number(max(a["rmtop4"]["over_refusal"] - a["clean"]["over_refusal"] for a in mixed), lead=True)}


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Table 20. Smallest repairing rank k* against the update's spectrum, Llama-3.1-8B.",
         ["f", "seed", "k*", "part. ratio", "top-2 energy"], table(args.records), args.out)
    print()
    for key, value in mixed_summary(args.records).items():
        print(f"{key:48s} {value}")
