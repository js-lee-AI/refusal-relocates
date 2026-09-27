"""The `refusal` command. `python -m refusal` runs the same thing."""

import argparse
import json
from pathlib import Path

from . import __version__


def parser():
    root = argparse.ArgumentParser(
        prog="refusal", description="Refusal localization, layer freezing and weight-space repair")
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = root.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="run the CPU quickstart")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--k", type=int, default=2)

    prepare = commands.add_parser("prepare", help="restore the frozen selection from public data")
    prepare.add_argument("--selection", default=None, help="defaults to the manifest shipped with the package")
    for source in ("pku", "alpaca", "advbench", "xstest"):
        prepare.add_argument("--" + source, nargs="+", required=True)
    prepare.add_argument("--out", default="data/prompt_sets.json")

    def common(name, help):
        command = commands.add_parser(name, help=help)
        command.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
        command.add_argument("--revision", default=None)
        command.add_argument("--device", default="cuda")
        command.add_argument("--data", default="data/prompt_sets.json")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--out", required=True)
        return command

    train = common("train", "fine-tune a LoRA adapter, optionally with a frozen band")
    train.add_argument("--n-train", type=int, default=100)
    train.add_argument("--rank", type=int, default=16)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--modules", choices=["attention", "all"], default="attention")
    train.add_argument("--freeze", choices=["none", "bottom", "top"], default="none")
    train.add_argument("--n-frozen", type=int, default=0)
    train.add_argument("--mode", choices=["standard", "spread", "projected"], default="standard")
    train.add_argument("--lam", type=float, default=1.)
    train.add_argument("--k", type=int, default=2)
    train.add_argument("--project-every", type=int, default=25)
    train.add_argument("--harm-fraction", type=float, default=None)
    train.add_argument("--benign-pool", choices=["general", "listy", "longform", "terse"])

    for name, help in (("evaluate", "score refusal before and after top-k removal"),
                       ("patch", "lockstep patching from the clean into the fine-tuned model")):
        command = common(name, help)
        command.add_argument("--adapter", required=True)
        command.add_argument("--eval-seed", type=int, default=42)
        command.add_argument("--n-eval", type=int, default=40 if name == "patch" else 100)
        command.add_argument("--batch-size", type=int, default=8 if name == "patch" else 16)
        command.add_argument("--max-new-tokens", type=int, default=192 if name == "patch" else 256)
        command.add_argument("--normalize-apostrophes", action="store_true")
        if name == "evaluate":
            command.add_argument("--ranks", default="2")
            command.add_argument("--n-xstest", type=int, default=50)
            command.add_argument("--random-control", action="store_true")
            command.add_argument("--task-utility", action="store_true")
        else:
            command.add_argument("--layers", default="")

    probe = common("probe", "fit a probe on the clean model and apply it frozen to the fine-tuned one")
    probe.add_argument("--adapter", required=True)
    probe.add_argument("--n-probe", type=int, default=200)

    spectra = commands.add_parser("spectra", help="spectral statistics of a saved adapter, on CPU")
    spectra.add_argument("--adapter", required=True)
    spectra.add_argument("--module-set", choices=["attention", "all"], default="attention")
    spectra.add_argument("--out", required=True)

    detector = commands.add_parser("detector", help="calibrate PR/rank at a benign false-positive budget")
    detector.add_argument("--benign", nargs="+", required=True, help="JSON outputs of the spectra command")
    detector.add_argument("--attacks", nargs="+", required=True)
    detector.add_argument("--budget", type=float, default=.05)
    detector.add_argument("--out", required=True)
    return root


def demo(seed=0, k=2):
    """Top-k removal on a toy rank-16 update, once concentrated and once spread flat."""
    import numpy as np

    from .spectrum import remove_top_k, update_stats

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((16, 512)) / 512 ** 0.5
    b = rng.standard_normal((512, 16)) / 16 ** 0.5
    lines = []
    for name, decay in (("ordinary", 0.5), ("spread", 1.0)):
        scaled = b * decay ** np.arange(16)
        before = update_stats(a, scaled)
        after = update_stats(*remove_top_k(a, scaled, k))
        lines.append(f"{name:8s}  PR/r {before['pr_over_r']:.2f}  top-2 energy "
                     f"{before['top2_energy_fraction']:.2f}  norm left after top-{k} removal "
                     f"{after['unscaled_norm'] / before['unscaled_norm']:.2f}")
    return "\n".join(lines)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "demo":
        print(demo(args.seed, args.k))
        return 0
    if args.command == "prepare":
        from .data import SELECTION, prepare, write_json
        data = prepare(args.selection or SELECTION,
                       {key: getattr(args, key) for key in ("pku", "alpaca", "advbench", "xstest")})
        write_json(args.out, data)
        return 0
    if args.command == "spectra":
        from .data import write_json
        from .spectral import adapter_stats
        write_json(args.out, adapter_stats(args.adapter, args.module_set))
        return 0
    if args.command == "detector":
        from .data import write_json
        from .spectrum import calibrate_detector
        benign = [json.loads(Path(p).read_text()) for p in args.benign]
        attacks = [json.loads(Path(p).read_text()) for p in args.attacks]
        if len({row["base_model"] for row in benign + attacks}) != 1:
            raise ValueError("Calibrate and evaluate the detector within one checkpoint")
        if len({row["module_set"] for row in benign + attacks}) != 1:
            raise ValueError("Detector scores must use the same projection set")
        result = calibrate_detector([row["pr_over_r"] for row in benign],
                                    [row["pr_over_r"] for row in attacks], args.budget)
        write_json(args.out, {**result, "attack_n": len(attacks)})
        return 0

    from .common import set_seed
    from .data import (assert_disjoint, benign_pools, load_data, mixture, shuffled_subset,
                       split_task_pool)
    set_seed(args.seed)
    data = load_data(args.data)
    if args.command == "train":
        from .train import train_adapter
        if args.benign_pool and args.harm_fraction is not None:
            raise ValueError("Choose a benign pool or a harmful fraction")
        if args.benign_pool:
            pool = benign_pools(data["alpaca_pairs"])[args.benign_pool]
            pairs = shuffled_subset(pool, args.n_train, args.seed)
        elif args.harm_fraction is not None:
            pairs = mixture(data, args.n_train, args.harm_fraction, args.seed)
        else:
            pairs = shuffled_subset(data["pku_unsafe"], args.n_train, args.seed)
        eval_sets = [data["advbench"], data["xstest_safe"]]
        if args.harm_fraction is not None:
            _, heldout = split_task_pool(data["alpaca_pairs"])
            eval_sets.append([row["prompt"] for row in heldout])
        assert_disjoint(pairs, eval_sets)
        train_adapter(args, pairs)
    elif args.command == "probe":
        from .probe import run_probe
        run_probe(args, data)
    elif args.command == "patch":
        from .evaluate import run_patching
        run_patching(args, data)
    elif args.command == "evaluate":
        from .evaluate import run_evaluation
        run_evaluation(args, data)
    return 0
