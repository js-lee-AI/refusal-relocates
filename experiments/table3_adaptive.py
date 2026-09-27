"""Table 3. Three dose-100 attacks on Llama-3.1-8B, evaluated on the same 50 AdvBench prompts.

Standard is LoRA on all seven projections for five epochs. Spread adds the concentration penalty
at lambda = 1. Projected removes the top two directions during training, and only its two landed
seeds enter. Norm and spectral statistics average all adapted modules.
"""

from _adaptive import adapters, landed, projected, spread, standard
from _records import parser, show, span


def sig3(x):
    text = f"{x:#.3g}"
    return text[1:] if text.startswith("0.") else text


def sig_span(values):
    values = list(values)
    lo, hi = sig3(min(values)), sig3(max(values))
    return lo if lo == hi else f"{lo}–{hi}"


ROWS = [("||dW||_F", "update_norm", sig_span), ("Part. ratio", "participation_ratio", sig_span),
        ("Top-2 energy", "top2_energy", sig_span), ("Refusal, no repair", "common_attacked", span),
        ("LG unsafe, no repair", "common_attacked_lg_unsafe", span),
        ("Refusal, after top-2", "common_repaired", span),
        ("LG unsafe, after top-2", "common_repaired_lg_unsafe", span)]


def table(records=None):
    rows = adapters(records)
    arms = [standard(rows), spread(rows), [r for r in projected(rows) if landed(r)]]
    return [[label] + [fmt(r[key] for r in arm) for arm in arms] for label, key, fmt in ROWS]


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Table 3. Standard, spread and projected attacks on Llama-3.1-8B at dose 100.",
         ["", "Standard 5 ep", "Spread lambda=1", "Projected k=2"], table(args.records), args.out)
