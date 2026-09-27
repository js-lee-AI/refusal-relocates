"""Shared helpers for the table scripts. They read the run records and format cells as the paper prints them."""

import argparse
import csv
import sys
from pathlib import Path

RECORDS = Path(__file__).resolve().parents[1] / "records"
RESULTS = Path(__file__).resolve().parents[1] / "results"

NAMES = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "allenai/Llama-3.1-Tulu-3-8B-DPO": "Tulu-3-8B-DPO",
    "allenai/OLMo-2-1124-7B-Instruct": "OLMo-2-7B",
    "allenai/OLMo-2-1124-13B-Instruct": "OLMo-2-13B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "01-ai/Yi-1.5-9B-Chat": "Yi-1.5-9B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B-v0.3",
}
# The order the paper's tables list the six checkpoints in.
SIX = ["meta-llama/Llama-3.1-8B-Instruct", "allenai/Llama-3.1-Tulu-3-8B-DPO", "allenai/OLMo-2-1124-7B-Instruct",
       "allenai/OLMo-2-1124-13B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "01-ai/Yi-1.5-9B-Chat"]


def _value(text):
    if text == "":
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def load(name, records=None):
    with open(Path(records or RECORDS) / name, newline="") as f:
        return [{k: _value(v) for k, v in row.items()} for row in csv.DictReader(f)]


def number(x, digits=2, lead=False):
    text = f"{x:.{digits}f}"
    return text if lead or not text.startswith("0.") else text[1:]


def span(values, digits=2, lead=False):
    """A seed range as the paper prints it, one number when the seeds agree at that precision."""
    values = list(values)
    if not values:
        return "n/a"
    lo, hi = number(min(values), digits, lead), number(max(values), digits, lead)
    return lo if lo == hi else f"{lo}–{hi}"


def show(title, header, rows, out=None):
    """Print a paper table as aligned plain text, and write it to `out` as CSV when given.
    A short row (a note) does not set the widths."""
    full = [r for r in [header] + rows if len(r) == len(header)]
    widths = [max(len(str(r[i])) for r in full) for i in range(len(header))]
    print(title)
    for r in [header] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip())
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows([header] + rows)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--records", default=str(RECORDS), help="folder with the run records (default: records/)")
    p.add_argument("--out", default=str(RESULTS / f"{Path(sys.argv[0]).stem}.csv"),
                   help="CSV file for the rebuilt table (default: results/<script>.csv)")
    return p


def select(rows, **match):
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def by_arm(rows, arms=("clean", "attack", "freeze", "matched"), **match):
    """Gate-passing freeze records of one cell, grouped by arm."""
    kept = [r for r in select(rows, **match) if r["gate"]]
    return {arm: [r for r in kept if r["arm"] == arm] for arm in arms}


def checkpoints():
    import json
    config = json.loads((Path(__file__).parent / "configs" / "models.json").read_text())
    return {c["model"]: c for c in config.values()}
