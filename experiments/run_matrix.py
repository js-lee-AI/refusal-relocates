import argparse
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path


def plan(suite, models, seeds):
    cells = []
    if suite == "dose":
        for model, seed, dose in itertools.product(models, seeds, [5, 10, 25, 50, 100]):
            cells.append((model, seed, f"dose{dose}", ["--n-train", str(dose)], ["probe", "patch", "evaluate"]))
    elif suite in {"freeze", "four_layers", "one_two_layers", "low_dose"}:
        low = {"llama": [5, 10, 25], "tulu": [25], "olmo7": [25],
               "olmo13": [5, 10], "qwen": [25], "yi": [10]}
        for model, seed in itertools.product(models, seeds):
            layers = models[model]["layers"]
            if suite == "four_layers":
                counts = [layers - 4]
            elif suite == "one_two_layers":
                counts = [layers - 2, layers - 1]
            else:
                counts = [models[model]["boundary"] + 1]
            for dose, count in itertools.product(low[model] if suite == "low_dose" else [100], counts):
                for side in ("none", "bottom", "top"):
                    if side == "none" and count != counts[0]:
                        continue  # one unrestricted attack serves both boundaries
                    # the freeze suite also patches its unrestricted arm, on the default grid
                    patch = suite == "freeze" and side != "top"
                    cells.append((model, seed, f"n{dose}_{side}{count}",
                                  ["--n-train", str(dose), "--freeze", side, "--n-frozen", str(count)],
                                  ["evaluate"] + (["patch"] if patch else [])))
    elif suite == "ladder":
        for seed, count in itertools.product(seeds, [0, 6, 12, 18, 24, 28]):
            cells.append(("llama", seed, f"bottom{count}",
                          ["--freeze", "bottom" if count else "none", "--n-frozen", str(count)],
                          ["evaluate", "patch"]))
    elif suite in {"adaptive", "epochs", "lambda"}:
        if suite == "adaptive":
            arms = [("standard", 5, "all", 0), ("spread", 5, "all", 1),
                    ("projected", 5, "all", 0)]
        elif suite == "epochs":
            arms = [("standard", ep, "all", 0) for ep in [1, 2, 3, 5, 8]]
            arms += [("standard", 3, "attention", 0), ("standard", 8, "attention", 0)]
        else:
            arms = [("spread", 5, "all", lam) for lam in [1., .5, .25, .1, .03, .01, .003]]
        for seed, (mode, epochs, modules, lam) in itertools.product(seeds, arms):
            cells.append(("llama", seed, f"{mode}_{modules}_ep{epochs}_lam{lam}",
                          ["--mode", mode, "--epochs", str(epochs), "--modules", modules,
                           "--lam", str(lam), "--project-every", "25"], ["evaluate"]))
    elif suite in {"mixed", "mixed_freeze"}:
        fractions = [.05, .15, .35] if suite == "mixed_freeze" else [0., .05, .15, .35, 1.]
        for seed, fraction in itertools.product(seeds, fractions):
            for side in ["none", "bottom", "top"] if suite == "mixed_freeze" else ["none"]:
                cells.append(("llama", seed, f"mix{fraction}_{side}",
                              ["--harm-fraction", str(fraction), "--freeze", side, "--n-frozen", "18"], ["evaluate"]))
    elif suite == "benign_detector":
        records = json.loads((Path(__file__).parent / "configs/benign_runs.json").read_text())
        for row in records:
            if row["seed"] not in seeds:
                continue
            pool = row["source"].removeprefix("alpaca_")
            cells.append(("llama", row["seed"], f"{pool}_r{row['rank']}_ep{row['epochs']}_rep{row['replicate']}",
                          ["--benign-pool", pool, "--rank", str(row["rank"]),
                           "--epochs", str(row["epochs"]), "--modules", "all"], []))
        for seed in seeds:
            cells.append(("llama", seed, "mixed_f0", ["--harm-fraction", "0"], []))
    return cells


def commands(args):
    models = json.loads((Path(__file__).parent / "configs/models.json").read_text())
    selected = {key: models[key] for key in args.models.split(",")}
    for model, seed, tag, train, analyses in plan(args.suite, selected, args.seeds):
        if model not in selected:
            continue
        directory = Path(args.out) / args.suite / f"{model}_s{seed}_{tag}"
        base = ["--model", models[model]["model"], "--seed", str(seed),
                "--data", args.data, "--device", args.device]
        yield [sys.executable, "-m", "refusal", "train", *base, *train, "--out", str(directory)]
        adapter = str(directory / "adapter")
        yield [sys.executable, "-m", "refusal", "spectra", "--adapter", adapter,
               "--out", str(directory / "spectrum.json")]
        for analysis in analyses:
            extra = []
            if analysis != "probe":
                common = args.suite in {"adaptive", "epochs", "lambda"}
                extra += ["--eval-seed", "42" if common else str(seed)]
                if common:
                    extra += ["--n-eval", "50", "--normalize-apostrophes"]
            if analysis == "patch" and args.suite == "ladder":
                extra += ["--layers", "6,9,12,15,18,21,24,27,28,29,30", "--n-eval", "100"]
            if analysis == "patch" and args.suite == "freeze" and "_bottom" in tag:
                boundary = models[model]["boundary"]
                layers = sorted(set(range(boundary + 1, min(boundary + 6, models[model]["layers"] - 1))))
                extra += ["--layers", ",".join(map(str, layers))]
                if model in {"tulu", "olmo7"}:
                    extra += ["--n-eval", "100"]
            if analysis == "evaluate" and args.suite in {"mixed", "mixed_freeze"}:
                extra += ["--task-utility", "--n-eval", "50", "--ranks", "1,2,4,8"]
            if analysis == "evaluate" and args.suite == "dose":
                extra += ["--random-control", "--ranks", "1,2,4", "--n-eval", "50"]
            result = directory / ("probe.json" if analysis == "probe" else analysis)
            yield [sys.executable, "-m", "refusal", analysis, *base, "--adapter", adapter,
                   "--out", str(result), *extra]


def main():
    parser = argparse.ArgumentParser(description="Print or run the main experiment commands")
    parser.add_argument("--suite", required=True, choices=["dose", "freeze", "four_layers", "one_two_layers", "low_dose",
                        "ladder", "adaptive", "epochs", "lambda", "mixed", "mixed_freeze", "benign_detector"])
    parser.add_argument("--models", default="llama,tulu,olmo7,olmo13,qwen,yi")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--data", default="data/prompt_sets.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    for command in commands(args):
        print(shlex.join(command), flush=True)
        if args.run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
