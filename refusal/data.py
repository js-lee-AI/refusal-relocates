import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path

# SHA-256 of every selected example, in order. Only hashes are shipped, never the text.
SELECTION = Path(__file__).resolve().parent / "selection.json"


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False) + "\n")


def fingerprint(value, kind):
    if kind == "pair":
        value = [value["prompt"].strip(), value["response"].strip()]
    else:
        value = value.strip()
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_rows(path):
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(newline="") as stream:
            yield from csv.DictReader(stream)
    elif path.suffix in {".jsonl", ".ndjson"}:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
    else:
        yield from json.loads(path.read_text())


def candidates(rows, source, kind):
    for row in rows:
        if source == "pku":
            prompt = (row.get("prompt") or "").strip()
            if kind == "prompt":
                yield prompt
            else:
                for side in ("0", "1"):
                    response = (row.get("response_" + side) or "").strip()
                    if prompt and response:
                        yield {"prompt": prompt, "response": response}
        elif source == "alpaca":
            prompt = (row.get("instruction") or "").strip()
            text_input = (row.get("input") or "").strip()
            if text_input:
                prompt += "\n" + text_input
            yield {"prompt": prompt, "response": (row.get("output") or "").strip()}
        elif source == "advbench":
            yield row["goal"].strip()
        elif source == "xstest":
            yield row["prompt"].strip()


def select_by_hash(rows, wanted, kind):
    needed = set(wanted)
    found = {}
    for row in rows:
        key = fingerprint(row, kind)
        if key in needed:
            found[key] = row
            if len(found) == len(needed):
                break
    missing = needed - found.keys()
    if missing:
        raise ValueError(f"The supplied source is missing {len(missing)} selected examples")
    return [found[key] for key in wanted]


def prepare(selection_path, sources):
    selection = json.loads(Path(selection_path).read_text())
    result = {}
    for key, spec in selection["sets"].items():
        source = spec["source"]
        if source not in sources:
            raise ValueError(f"Missing {source} source")
        paths = sources[source]
        rows = (r for path in paths for r in read_rows(path))
        result[key] = select_by_hash(candidates(rows, source, spec["kind"]),
                                    spec["sha256"], spec["kind"])
    return result


def validate_data(data, selection_path):
    selection = json.loads(Path(selection_path).read_text())
    for key, spec in selection["sets"].items():
        actual = [fingerprint(row, spec["kind"]) for row in data[key]]
        if actual != spec["sha256"]:
            raise ValueError(f"Selection or ordering differs for {key}")


def load_data(path):
    data = json.loads(Path(path).read_text())
    validate_data(data, SELECTION)
    return data


def shuffled_subset(rows, n, seed):
    if not 0 <= n <= len(rows):
        raise ValueError("Requested sample exceeds the available data")
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def split_task_pool(pairs, n_hold=200):
    unique = {}
    for pair in pairs:
        if pair["prompt"].strip() and pair["response"].strip():
            unique.setdefault(pair["prompt"].strip(), pair)
    rows = list(unique.values())
    random.Random(0).shuffle(rows)
    if not 0 < n_hold < len(rows):
        raise ValueError("Invalid task holdout size")
    return rows[:-n_hold], rows[-n_hold:]


def mixture(data, n, fraction, seed):
    if not 0 <= fraction <= 1:
        raise ValueError("Harmful fraction must lie in [0, 1]")
    n_harm = round(n * fraction)
    harmful = shuffled_subset(data["pku_unsafe"], n_harm, seed)
    task_train, _ = split_task_pool(data["alpaca_pairs"])
    benign = random.Random(seed).sample(task_train, n - n_harm)
    rows = harmful + benign
    random.Random(seed + 1).shuffle(rows)
    return rows


def benign_pools(pairs, size=125):
    ids = list(range(len(pairs)))
    words = [len(p["response"].split()) for p in pairs]
    pattern = re.compile(r"^\s*(?:\d+[.)]|[-*\u2022])\s+", re.M)
    bullets = [len(pattern.findall(p["response"])) for p in pairs]
    general = sorted(random.Random(0).sample(ids, size))
    rest = [i for i in ids if i not in set(general)]
    listy = sorted(rest, key=lambda i: (-bullets[i], -words[i], i))[:size]
    rest = [i for i in rest if i not in set(listy)]
    longform = sorted(rest, key=lambda i: (-words[i], i))[:size]
    rest = [i for i in rest if i not in set(longform)]
    terse = sorted(rest, key=lambda i: (words[i], i))[:size]
    result = dict(general=general, listy=listy, longform=longform, terse=terse)
    if any(len(v) != size for v in result.values()):
        raise ValueError("Too few examples for disjoint benign pools")
    return {key: [pairs[i] for i in indices] for key, indices in result.items()}


def assert_disjoint(pairs, prompt_sets):
    train = {row["prompt"].strip() for row in pairs}
    if any(train & {p.strip() for p in prompts} for prompts in prompt_sets):
        raise ValueError("Training and evaluation prompts overlap")
