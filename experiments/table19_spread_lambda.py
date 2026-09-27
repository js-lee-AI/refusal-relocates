"""Table 19. Sweeping the concentration penalty lambda on the spread attacker.

Llama-3.1-8B at dose 100, rank 16, three seeds at the two ends of the sweep and two elsewhere,
each seed scored on its own 50 AdvBench prompts. The score here is the participation ratio over
rank, averaged over all adapted modules.
"""

from _adaptive import adapters
from _records import parser, show, span

LAMBDAS = [1.0, 0.5, 0.25, 0.1, 0.03, 0.01, 0.003]


def table(records=None):
    rows = adapters(records)
    out = []
    for lam in LAMBDAS:
        rs = [r for r in rows if r["mode"] == "spread" and r["lam"] == lam]
        out.append([str(lam), span((r["participation_ratio"] / 16 for r in rs), 3, lead=True),
                    span((r["update_norm"] for r in rs), 3, lead=True),
                    span((r["own_repaired"] for r in rs), lead=True)])
    return out


def ranges(records=None):
    """Clean and attacked refusal over the whole sweep."""
    rs = [r for r in adapters(records) if r["mode"] == "spread"]
    return span((r["own_clean"] for r in rs), lead=True), span((r["own_attacked"] for r in rs), lead=True)


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Table 19. The spread attacker across lambda, Llama-3.1-8B, dose 100.",
         ["lambda", "detector score", "||dW||_F", "refusal after removal"], table(args.records), args.out)
    clean, attacked = ranges(args.records)
    print(f"clean refusal {clean}, refusal after the attack {attacked}")
