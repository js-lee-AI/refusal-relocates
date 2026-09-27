"""Table 14. The band freeze against a mixed adapter.

100 training examples of which a fraction f is harmful, Llama-3.1-8B with [0, 17] frozen, three
seeds, ranges over seeds. Task is the mean negative log-likelihood of 200 held-out references.
"""

from _records import load, parser, select, show, span

LABEL = {"attack": "unrestricted", "freeze": "[0, 17] frozen", "matched": "matched"}


def cells(rows):
    kept = [r for r in rows if r["gate"]]
    return [span([r["refusal"] for r in kept], lead=True), span([r["lg_unsafe"] for r in kept], lead=True),
            span([r["over_refusal"] for r in kept], lead=True), span([r["task_nll"] for r in kept], 3, lead=True)]


def table(records=None):
    rows = load("mixed_freeze.csv", records)
    out = [["Clean"] + cells(select(rows, arm="clean"))]
    for f in sorted({r["harm_fraction"] for r in rows}):
        for arm in ("attack", "freeze", "matched"):
            out.append([f"f={f:.2f}, {LABEL[arm]}"] + cells(select(rows, harm_fraction=f, arm=arm)))
    return out


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Table 14. Refusal, Llama-Guard unsafe, over-refusal and task NLL for a mixed adapter.",
         ["f and arm", "refusal", "LG unsafe", "over-refusal", "task NLL"], table(args.records), args.out)
