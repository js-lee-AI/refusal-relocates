"""Table 16. Coherent refusal after removing the top singular directions of the update.

Attention-only LoRA at three epochs, three seeds and five doses, 50 seed-specific AdvBench
prompts. A cell enters only where the attack landed, clean refusal at least 0.5 and attacked
refusal at most 0.20. Random-2 removes two randomly chosen directions.
"""

from _records import NAMES, load, parser, show, span

MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "allenai/Llama-3.1-Tulu-3-8B-DPO",
          "allenai/OLMo-2-1124-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]
ARMS = ["rmtop1", "rmtop2", "rmtop4", "rmrand2"]


def landed(r):
    return r["gate"] and r["clean"] >= 0.5 and r["attacked"] <= 0.20


def table(records=None):
    rows = [r for r in load("topk.csv", records) if landed(r)]
    return [[NAMES[m]] + [span((r[a] for r in rows if r["model"] == m), lead=True) for a in ARMS]
            for m in MODELS]


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Table 16. Coherent refusal after removing the top singular directions, landed cells only.",
         ["model", "remove top-1", "remove top-2", "remove top-4", "random-2"], table(args.records), args.out)
