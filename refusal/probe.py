import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .common import get_layers, last_token_acts, load_model, release_memory, write_json


def cv_auroc(x, y, seed):
    scores = []
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        probe = LogisticRegression(max_iter=2000, C=1.).fit(x[train], y[train])
        scores.append(roc_auc_score(y[test], probe.decision_function(x[test])))
    return float(np.mean(scores))


def at_threshold(harmful, benign, threshold):
    return {"threshold": float(threshold),
            "accuracy": float(np.concatenate([harmful > threshold, benign <= threshold]).mean()),
            "tpr": float((harmful > threshold).mean()),
            "fpr": float((benign > threshold).mean())}


def compare_features(clean, attacked, y, seed):
    probe = LogisticRegression(max_iter=2000, C=1.).fit(clean, y)
    scores = probe.decision_function(attacked)
    harmful, benign = scores[y == 1], scores[y == 0]
    permutation = np.random.RandomState(seed).permutation(len(benign))
    calibration, test = np.split(permutation, [len(benign) // 2])
    threshold = np.quantile(benign[calibration], .95)
    return {"clean_cv_auroc": cv_auroc(clean, y, seed),
            "attacked_cv_auroc": cv_auroc(attacked, y, seed),
            "frozen_auroc": float(roc_auc_score(y, scores)),
            "frozen_accuracy": float(((scores > 0) == y).mean()),
            "fixed_threshold_heldout": at_threshold(harmful, benign[test], 0),
            "recalibrated_heldout": at_threshold(harmful, benign[test], threshold)}


def run_probe(args, data):
    n = args.n_probe
    harmful = data["pku_harm_intent"][200:200 + n]
    benign = data["pku_benign_intent"][200:200 + n]
    if len(harmful) != n or len(benign) != n:
        raise ValueError("Probe contrast is too small")
    prompts = harmful + benign
    y = np.array([1] * n + [0] * n)
    clean, tokenizer = load_model(args.model, args.device, revision=args.revision)
    layers = [int(len(get_layers(clean)) * f) for f in (.3, .45, .6, .75)]
    clean_features = {layer: last_token_acts(clean, tokenizer, prompts, layer) for layer in layers}
    del clean
    release_memory()
    attacked, tokenizer = load_model(args.model, args.device, args.adapter, args.revision)
    result = {layer: compare_features(clean_features[layer],
                                     last_token_acts(attacked, tokenizer, prompts, layer), y, args.seed)
              for layer in layers}
    write_json(args.out, {"config": vars(args), "probes": result})
