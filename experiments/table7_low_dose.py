"""Table 7. The band freeze below its breaking dose.

Each checkpoint is shown at the low dose where its unrestricted attack lands and passes the gate.
On OLMo-2-13B the attack does not land at 5 or 10 examples, so every seed fails the gate.
"""

from _records import NAMES, by_arm, checkpoints, load, parser, show, span

ARMS = ["clean", "attack", "freeze", "matched"]
DOSES = [("meta-llama/Llama-3.1-8B-Instruct", [5]), ("allenai/OLMo-2-1124-7B-Instruct", [25]),
         ("allenai/Llama-3.1-Tulu-3-8B-DPO", [25]), ("Qwen/Qwen2.5-14B-Instruct", [25]),
         ("01-ai/Yi-1.5-9B-Chat", [10]), ("allenai/OLMo-2-1124-13B-Instruct", [5, 10])]


def table(records=None):
    rows = load("freeze.csv", records)
    config = checkpoints()
    out = []
    for model, doses in DOSES:
        cells = [by_arm(rows, model=model, frozen_upto=config[model]["boundary"], dose=d) for d in doses]
        if not any(c["attack"] for c in cells):
            out.append([NAMES[model], ", ".join(map(str, doses)),
                        "no successful unrestricted attack at either dose"])
            continue
        cell = cells[0]
        out.append([NAMES[model], doses[0]] + [span(r["refusal"] for r in cell[a]) for a in ARMS])
    return out


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Table 7. Coherent refusal on AdvBench at low dose, ranges over the seeds that pass the gate.",
         ["model", "dose", "clean", "attack", "freeze", "matched"], table(args.records), args.out)
